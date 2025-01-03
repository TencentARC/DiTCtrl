from functools import partial
import math
from einops import rearrange, repeat
import numpy as np

import os
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from omegaconf import ListConfig

import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image

from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.mpu.layers import ColumnParallelLinear
from sgm.util import instantiate_from_config

from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
from sat.ops.layernorm import LayerNorm, RMSNorm
def concatenate_frames(frames):
    """
    Concatenate multiple frames horizontally into a single image
    
    Args:
        frames: List of frames (F, H, W)
        
    Returns:
        Concatenated image (H, F*W)
    """
    # Ensure all frames are of the same size
    H, W = frames[0].shape
    num_frames = len(frames)
    
    # Create a blank canvas
    concat_image = torch.zeros((H, W * num_frames), device=frames[0].device, dtype=frames[0].dtype)
    
    # Horizontally concatenate all frames
    for i, frame in enumerate(frames):
        concat_image[:, i*W:(i+1)*W] = frame
        
    return concat_image

class AttentionMapController:
    def __init__(self, save_dir, text_length, height, width, compressed_num_frames, thres=0.1):
        self.save_dir = save_dir
        self.text_length = text_length
        self.height = height
        self.width = width
        self.compressed_num_frames = compressed_num_frames
        self.thres = thres
        self.cross_attn_sum = None
        self.cross_attn_count = 0
        self.video_self_attn_sum = None
        self.video_self_attn_count = 0
        self.self_attn_mask = None
        self.text_self_attn_sum = None
        self.text_self_attn_count = 0
        print(f"Set mask save directory for AttentionMapController: {save_dir}")

    def get_self_attn_mask(self):
        return self.self_attn_mask
    def get_cross_attn_sum(self):
        return self.cross_attn_sum
    def get_cross_attn_count(self):
        return self.cross_attn_count
    
    def set_self_attn_mask(self, mask):
        self.self_attn_mask = mask

    def reset_cross_attns(self):
        self.cross_attn_sum = None
        self.cross_attn_count = 0

    def reset_self_attn_mask(self):
        self.self_attn_mask = None
    
    def save_cur_attn_map(self, q, k, cur_step, layer_id):
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize result tensor (on GPU)
        attn_map_mean = torch.zeros(batch_size, seq_len, seq_len, device=q.device, dtype=q.dtype)
        
        # Parameters for batch computation
        batch_chunk = 1  # Number of batches to process at a time
        head_chunk = 1   # Number of attention heads to process at a time, can be adjusted based on GPU memory
        for i in range(0, batch_size, batch_chunk):
            for j in range(0, num_heads, head_chunk):
                # Select data for current batch and attention heads
                q_chunk = q[i:i+batch_chunk, j:j+head_chunk]
                k_chunk = k[i:i+batch_chunk, j:j+head_chunk]
                
                # Calculate attention scores for current attention heads
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_probs = torch.softmax(attn_scores, dim=-1)
                
                # Add to mean attention map
                attn_map_mean[i:i+batch_chunk] += attn_probs.sum(dim=1)
        
        # Calculate mean
        attn_map_mean /= num_heads
        video_to_text_attn = attn_map_mean[:, self.text_length:, :self.text_length]

        # Update accumulated sum and count
        if self.cross_attn_sum is None:
            self.cross_attn_sum = video_to_text_attn
        else:
            self.cross_attn_sum += video_to_text_attn
        self.cross_attn_count += 1
        
        # Process video self attention
        video_to_video_attn = attn_map_mean[:, self.text_length:, self.text_length:]
        if self.video_self_attn_sum is None:
            self.video_self_attn_sum = video_to_video_attn
        else:
            self.video_self_attn_sum += video_to_video_attn
        self.video_self_attn_count += 1

        text_to_text_attn = attn_map_mean[:, :self.text_length, :self.text_length]
        if self.text_self_attn_sum is None:
            self.text_self_attn_sum = text_to_text_attn
        else:
            self.text_self_attn_sum += text_to_text_attn
        self.text_self_attn_count += 1

    def aggregate_cross_attn_map(self, token_idx=[1]):
        if self.cross_attn_sum is None or self.cross_attn_count == 0:
            return None

        attn_map = self.cross_attn_sum / self.cross_attn_count
        
        B, HWF, T = attn_map.shape
        F = HWF // (self.height * self.width)
        attn_map = attn_map.reshape(B, F, self.height, self.width, T)

        if isinstance(token_idx, (list, ListConfig)):
            attn_map = attn_map[..., token_idx]
            attn_map = attn_map.sum(dim=-1)  # Sum over selected tokens
        else:
            attn_map = attn_map[..., token_idx:token_idx+1].squeeze(-1)

        # Get min and max values using PyTorch
        attn_min = attn_map.amin(dim=(2, 3), keepdim=True)
        attn_max = attn_map.amax(dim=(2, 3), keepdim=True)
        
        normalized_attn = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)  # Add small value to avoid division by zero
        
        return normalized_attn
    
    def visualize_video_video_spatial_attn_map(self, step_idx, layer_idx, frame_idx=None,
                                   show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize spatial attention map (H*W, H*W)
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            frame_idx: int or None, specified frame index
            show_title: bool, whether to show title
            show_labels: bool, whether to show axis labels
            colormap: str, colormap to use
        """
        
        if self.video_self_attn_sum is None or self.video_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.video_self_attn_sum / self.video_self_attn_count
        B, HWF, HWF = attn_map.shape
        F = HWF // (self.height * self.width)
        HW = self.height * self.width
        
        # Reshape to [B, F, HW, F, HW]
        attn_map = attn_map.reshape(B, F, HW, F, HW)

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)

        # Determine frames to process
        if frame_idx is not None:
            frames_to_process = [frame_idx]
        else:
            frames_to_process = range(F)

        # Create attention map visualization for each frame
        for f_idx in frames_to_process:
            # Get spatial attention for current frame [HW, HW]
            spatial_attn = attn_map[0, f_idx, :, f_idx, :]
            
            # Normalize
            attn_min = spatial_attn.min()
            attn_max = spatial_attn.max()
            normalized_attn = (spatial_attn - attn_min) / (attn_max - attn_min + 1e-6)

            # Create figure
            plt.figure(figsize=(10, 10))
            plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)
            plt.colorbar()
            
            if show_labels:
                plt.xlabel('Spatial Position')
                plt.ylabel('Spatial Position')
            else:
                plt.xticks([])
                plt.yticks([])
            
            if show_title:
                plt.title(f'Spatial Attention Map (Frame {f_idx})')
            # Save image
            save_name = f'spatial_attn_map_frame{f_idx}.png'
            plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
            plt.close()

            # Optional: save as numpy array for later analysis
            np.save(os.path.join(save_path, f'spatial_attn_map_frame{f_idx}.npy'), 
                    normalized_attn.cpu().numpy())
            
    def visualize_text_attn_map(self, step_idx, layer_idx,
                               show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize text-to-text attention map
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            show_title: bool, whether to show title
            show_labels: bool, whether to show labels
            colormap: str, colormap to use (e.g. 'viridis', 'magma', etc.)
        """
        if not hasattr(self, 'text_self_attn_sum') or self.text_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.text_self_attn_sum / self.text_self_attn_count
        
        # Extract attention map from first batch
        text_attn = attn_map[1]  # [T, T]
        
        # Normalize
        attn_min = text_attn.min()
        attn_max = text_attn.max()
        normalized_attn = (text_attn - attn_min) / (attn_max - attn_min + 1e-6)

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)

        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)  # Use provided colormap
        plt.colorbar()
        
        if show_labels:
            plt.xlabel('Text Token Index')
            plt.ylabel('Text Token Index')
            plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        else:
            plt.xticks([])
            plt.yticks([])
        
        if show_title:
            plt.title(f'Text-to-Text Attention Map (Step {step_idx}, Layer {layer_idx})')
        
        # Save image
        save_name = f'text_attn_map.png'
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
        plt.close()

        # Save numpy array
        np.save(os.path.join(save_path, f'text_attn_map.npy'), 
                normalized_attn.cpu().numpy())
        
        return normalized_attn
    def visualize_video_video_temporal_attn_map(self, step_idx, layer_idx, token_h, token_w,
                                    show_title=True, show_labels=True, colormap='viridis'):
        """
        Visualize frame-to-frame attention map (F, F) at specified spatial position
        
        Args:
            step_idx: int, current step
            layer_idx: int, current layer
            token_h: int, token height position
            token_w: int, token width position
        """
        if self.video_self_attn_sum is None or self.video_self_attn_count == 0:
            return

        # Get mean attention map
        attn_map = self.video_self_attn_sum / self.video_self_attn_count
        B, HWF, HWF = attn_map.shape
        F = HWF // (self.height * self.width)
        HW = self.height * self.width
        
        # Reshape to [B, F, HW, F, HW]
        attn_map = attn_map.reshape(B, F, HW, F, HW)
        
        # Calculate token index in HW
        token_idx = token_h * self.width + token_w
        
        # Extract frame-to-frame attention at this position [F, F]
        temporal_attn = attn_map[0, :, token_idx, :, token_idx]
        
        # Normalize
        attn_min = temporal_attn.min()
        attn_max = temporal_attn.max()
        normalized_attn = (temporal_attn - attn_min) / (attn_max - attn_min + 1e-6)

        # Create save directory
        save_path = os.path.join(self.save_dir, f"step{step_idx}_layer{layer_idx}")
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 8))
        plt.imshow(normalized_attn.cpu().numpy(), cmap=colormap)
        plt.colorbar()
        
        if show_labels:
            plt.xlabel('Frame Index')
            plt.ylabel('Frame Index')
        else:
            plt.xticks([])
            plt.yticks([])
        
        if show_title:
            plt.title(f'Temporal Attention Map at Position ({token_h}, {token_w})')
            
        # Save image
        save_name = f'temporal_attn_map_pos_{token_h}_{token_w}.png'
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
        plt.close()

        # Save numpy array
        np.save(os.path.join(save_path, f'temporal_attn_map_pos_{token_h}_{token_w}.npy'), 
                normalized_attn.cpu().numpy())
        
        return normalized_attn   

    def visualize_token_attention_masks(self, token_indices, step, layer, frame_indices, num_tokens_to_show=6, layout='vertical'):
        """
        Generate and save token attention visualization results directly
        
        Args:
            token_indices: List[List[int]] - List of token indices, e.g. [[0], [1], [2], [0,1,2]]
            step: int - Current step
            layer: int - Current layer
            frame_indices: list - List of frame indices to visualize
            num_tokens_to_show: int - Number of tokens to display
            layout: str - Layout mode, 'horizontal' or 'vertical'
        """
        def apply_magma_colormap(image):
            """
            Convert grayscale image to magma colormap
            """
            img_array = np.array(image)
            img_normalized = img_array / 255.0
            colored_array = plt.cm.magma(img_normalized)
            colored_image = Image.fromarray((colored_array[:, :, :3] * 255).astype(np.uint8))
            return colored_image
        
            
        # Only process specified number of tokens
        token_indices = token_indices[:num_tokens_to_show]
        num_tokens = len(token_indices)
        
        for frame_idx in frame_indices:
            print(f"Processing step {step}, layer {layer}, frame {frame_idx}")
            
            # Create image layout
            if layout == 'horizontal':
                fig, axes = plt.subplots(1, num_tokens, figsize=(2*num_tokens, 2))
                subplot_adjust = {'wspace': 0.05, 'hspace': 0}
            else:  # vertical
                fig, axes = plt.subplots(num_tokens, 1, figsize=(2, 1.2*num_tokens))
                subplot_adjust = {'wspace': 0, 'hspace': 0.01}
            
            if num_tokens == 1:
                axes = [axes]
                
            # Set background color
            fig.patch.set_facecolor('white')
            
            # Generate and display attention map for each token/token combination
            for idx, token_idx in enumerate(token_indices):
                # Generate mask
                mask = self.aggregate_cross_attn_map(token_idx=token_idx)
                mask_target = mask[-1]  # (F, H, W)
                
                # Extract specified frame
                frame_mask = mask_target[frame_idx]  # (H, W)
                
                # Convert to PIL Image and apply magma colormap
                frame_img = Image.fromarray((frame_mask.cpu().numpy() * 255).astype(np.uint8))
                colored_frame = apply_magma_colormap(frame_img)
                
                # Display image
                axes[idx].imshow(np.array(colored_frame))
                axes[idx].axis('off')
                
            # Adjust subplot spacing
            plt.subplots_adjust(**subplot_adjust)
            
            # Tight layout
            if layout == 'vertical':
                plt.tight_layout(pad=0.1, h_pad=0.1)
                
            # Save image
            save_path = os.path.join(self.save_dir, f"step{step}_layer{layer}")
            os.makedirs(save_path, exist_ok=True)
            
            layout_str = 'horizontal' if layout == 'horizontal' else 'vertical'
            save_path = os.path.join(
                save_path,
                f'token_attention_{layout_str}_'
                f'step_{step}_layer_{layer}_'
                f'selected_frame_{frame_idx}_'
                f'first_{num_tokens_to_show}_tokens.png'
            )
            
            plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0, facecolor='white')
            plt.close()
            
            print(f"Token attention visualization saved to {save_path}")

class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"]  # (b,t,c,h,w)
        B, T = images.shape[:2]
        emb = images.view(-1, *images.shape[2:])
        emb = self.proj(emb)  # ((b t),d,h/2,w/2)
        emb = emb.view(B, T, *emb.shape[1:])
        emb = emb.flatten(3).transpose(2, 3)  # (b,t,n,d)
        emb = rearrange(emb, "b t n d -> b (t n) d")

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d)

        emb = emb.contiguous()
        return emb  # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_height * grid_width, axis=1)  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    return pos_embed  # [T, H*W, D]


def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.text_length = text_length
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)), requires_grad=False
        )
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs["images"].shape[1] == 1:
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
    ):
        super().__init__()
        self.rot_v = rot_v
        self.text_length = text_length

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        freqs = rearrange(freqs, "t h w d -> (t h w) d")

        freqs = freqs.contiguous()
        freqs_sin = freqs.sin()
        freqs_cos = freqs.cos()
        self.register_buffer("freqs_sin", freqs_sin)
        self.register_buffer("freqs_cos", freqs_cos)

    def rotary(self, t, **kwargs):
        seq_len = t.shape[2]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]

        query_layer[:, :, self.text_length :] = self.rotary(query_layer[:, :, self.text_length :])
        key_layer[:, :, self.text_length :] = self.rotary(key_layer[:, :, self.text_length :])
        if self.rot_v:
            value_layer[:, :, self.text_length :] = self.rotary(value_layer[:, :, self.text_length :])

        return attention_fn_default(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum("nlpqc->ncplq", x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True))

        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)

        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)
            ]
        )

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args["layer_id"]].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args["layer_id"]](x)
        hidden = origin.activation_func(x2) * x1
        x = origin.dense_4h_to_h(hidden)
        return x


class BaseAdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        text_length,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        start_step=2,
        start_layer=5,
        layer_idx=None,
        step_idx=None,
        overlap_size=6,
        sampling_num_frames=13,
        end_step=50,        
        end_layer=30,      
        mask_save_dir=None,
        ref_token_idx=None,
        cur_token_idx=None,
        attn_map_step_idx=None,
        attn_map_layer_idx=None,
        thres=0.1,
        num_prompts=None,
        num_transition_blocks=None,
        longer_mid_segment=None,
        is_edit=False,
    ):
        super().__init__()
        print("BaseAdaLNMixin Init")
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.text_length = text_length

        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
        self.is_edit = is_edit
        self.cur_step = 0
        self.cur_layer = 0 
        self.num_prompts = num_prompts
        self.num_transition_blocks = num_transition_blocks
        self.longer_mid_segment = longer_mid_segment
        self.count_segment = 0
        self.end_step = end_step
        self.end_layer = end_layer
        self.start_step = start_step
        self.start_layer = start_layer
        self.overlap_size = overlap_size
        self.sampling_num_frames = sampling_num_frames
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))

        self.attn_map_step_idx = attn_map_step_idx if attn_map_step_idx is not None else [5,10,15,20,25,30,35,40,45,50]
        self.attn_map_layer_idx = attn_map_layer_idx if attn_map_layer_idx is not None else [5,10,15,20,25,30]

        self.thres = thres

        self.mask_save_dir = mask_save_dir

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)

        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)

        attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)

        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)
        self.cur_layer += 1
        if self.cur_layer == self.num_layers:
            self.cur_layer = 0
            if self.is_edit:  # ever step, we just use two segments
                self.cur_step += 1
            else:  # every step, we use very long segments
                self.count_segment += 1
                # if self.count_segment == 2 * (self.num_prompts - 1) :
                total_segments = self.num_prompts + self.num_transition_blocks * (self.num_prompts - 1) + self.longer_mid_segment * (self.num_prompts - 2)
                # Every calculation, we use two segments, and stride=1. when equals, we reachs the last segment combination of one step
                if self.count_segment == total_segments - 1: 
                    self.count_segment = 0
                    self.cur_step += 1
            self.after_total_layers()
        return hidden_states

    def reinit(self, parent_model=None):
        self.cur_step = 0
        self.cur_layer = 0
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict 
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)
            
        attn_output = old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )
        return attn_output

    def after_total_layers(self):
        pass

class VisualizeAdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        text_length,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        start_step=2,
        start_layer=5,
        layer_idx=None,
        step_idx=None,
        overlap_size=6,
        sampling_num_frames=13,
        end_step=50,        
        end_layer=30,      
        mask_save_dir=None,
        ref_token_idx=None,
        cur_token_idx=None,
        attn_map_step_idx=None,
        attn_map_layer_idx=None,
        thres=0.1,
        num_prompts=None,
        num_transition_blocks=None,
        longer_mid_segment=None,
        is_edit=False,
    ):
        super().__init__()
        print("VisualizeAdaLNMixin Init")
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.text_length = text_length

        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
        self.is_edit = is_edit
        self.cur_step = 0
        self.cur_layer = 0 
        self.num_prompts = num_prompts
        self.num_transition_blocks = num_transition_blocks
        self.longer_mid_segment = longer_mid_segment
        self.count_segment = 0
        self.end_step = end_step
        self.end_layer = end_layer
        self.start_step = start_step
        self.start_layer = start_layer
        self.overlap_size = overlap_size
        self.sampling_num_frames = sampling_num_frames
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))

        self.attn_map_step_idx = attn_map_step_idx if attn_map_step_idx is not None else [5,10,15,20,25,30,35,40,45,50]
        self.attn_map_layer_idx = attn_map_layer_idx if attn_map_layer_idx is not None else [5,10,15,20,25,30]

        self.thres = thres

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)
            self.attn_controller = AttentionMapController(self.mask_save_dir, self.text_length, self.height, self.width, self.compressed_num_frames, 
                                                      thres=self.thres)

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)

        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)

        attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)

        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)
        self.cur_layer += 1
        if self.cur_layer == self.num_layers:
            self.cur_layer = 0
            self.cur_step += 1
            self.after_total_layers()

        return hidden_states

    def reinit(self, parent_model=None):
        self.cur_step = 0
        self.cur_layer = 0
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict 
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)
        
        self.attn_controller.save_cur_attn_map(query_layer, key_layer, cur_step=self.cur_step, layer_id=self.cur_layer)

        vis_params = {
            'show_title': True,  # Show title
            'show_labels': True,  # Show labels
            'colormap': 'viridis'  # Use viridis colormap
        }
        # hardcode here
        step_idx = [25]
        layer_idx = [15]
        frame_indices = [
            self.compressed_num_frames // 2,  # middle frame
        ]
        spatial_positions = [
            (self.height // 2, self.width // 2),     # Center position (30,45) -> (15, 22)
        ]  
        
        if self.cur_step in step_idx and self.cur_layer in layer_idx:
            # text-video attn/video-text attn
            self.attn_controller.visualize_token_attention_masks(
                token_indices=[[0],[1],[2],[3],[4],[5],[0,1,2,3,4,5]], 
                step=self.cur_step,
                layer=self.cur_layer,
                frame_indices=[10],
                num_tokens_to_show=6,
                layout='vertical'
            )
            
            # Text-text attn maps
            self.attn_controller.visualize_text_attn_map(self.cur_step, self.cur_layer, **vis_params)
            
            # Video-video spatial attention maps
            for frame_idx in frame_indices:
                self.attn_controller.visualize_video_video_spatial_attn_map(
                    self.cur_step, 
                    self.cur_layer,
                    frame_idx=frame_idx,
                    **vis_params
                )
            # Video-video temporal attention maps
            for pos_h, pos_w in spatial_positions:
                self.attn_controller.visualize_video_video_temporal_attn_map(
                    self.cur_step, 
                    self.cur_layer,
                    token_h=pos_h, 
                    token_w=pos_w,
                    **vis_params    
                )

        attn_output = old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )
        return attn_output

    def after_total_layers(self):
        pass



class KVSharingAdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        text_length,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        start_step=2,
        start_layer=5,
        layer_idx=None,
        step_idx=None,
        overlap_size=6,
        sampling_num_frames=13,
        end_step=50,        
        end_layer=30,      
        mask_save_dir=None,
        ref_token_idx=None,
        cur_token_idx=None,
        attn_map_step_idx=None,
        attn_map_layer_idx=None,
        thres=0.1,
        num_prompts=None,
        num_transition_blocks=None,
        longer_mid_segment=None,
        is_edit=False,
    ):
        super().__init__()
        print("KVSharingAdaLNMixin Init")
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.text_length = text_length
        
        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
        self.is_edit = is_edit
        self.num_prompts = num_prompts
        self.num_transition_blocks = num_transition_blocks
        self.longer_mid_segment = longer_mid_segment
        self.count_segment = 0

        self.cur_step = 0
        self.cur_layer = 0

        self.end_step = end_step
        self.end_layer = end_layer
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.overlap_size = overlap_size
        self.sampling_num_frames = sampling_num_frames
        
    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)
        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)
        
        # text_attention_input[-1] = text_attention_input[-1] + 0.2

        attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)

        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        # text_gate_msa = text_gate_msa + 4
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)

        self.cur_layer += 1
        if self.cur_layer == self.num_layers:
            self.cur_layer = 0
            if self.is_edit:  # ever step, we just use two segments
                self.cur_step += 1
            else:  # every step, we use very long segments
                self.count_segment += 1
                # if self.count_segment == 2 * (self.num_prompts - 1) :    #previous version
                total_segments = self.num_prompts + self.num_transition_blocks * (self.num_prompts - 1) + self.longer_mid_segment * (self.num_prompts - 2)
                # Every calculation, we use two segments, and stride=1. when equals, we reachs the last segment combination of one step
                if self.count_segment == total_segments - 1: 
                    self.count_segment = 0
                    self.cur_step += 1
            self.after_total_layers()

        return hidden_states

    def reinit(self, parent_model=None):
        self.cur_step = 0
        self.cur_layer = 0
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict 
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        if self.cur_step in self.step_idx and self.cur_layer in self.layer_idx:
            qu_s, qu_t, qc_s, qc_t = query_layer.chunk(4)
            ku_s, ku_t, kc_s, kc_t = key_layer.chunk(4)
            vu_s, vu_t, vc_s, vc_t = value_layer.chunk(4)

            # source branch
            out_u_s = old_impl(qu_s, ku_s, vu_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            out_c_s = old_impl(qc_s, kc_s, vc_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            # target branch
            out_u_t = self.attn_batch(qu_t, ku_s, vu_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            out_c_t = self.attn_batch(qc_t, kc_s, vc_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            
            out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)
            return out    
            
        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )

    
    def attn_batch(self, q, k, v, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs):
        
        # Ensure the input tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        '''Pure Version Will result in OOM error, as mentioned in the SwissArmyTransformer Link:
         https://github.com/THUDM/SwissArmyTransformer/blob/b68c4460b9fa2b5312be49a1da152986c6351262/sat/transformer_defaults.py#L60'''
        # Use PyTorch's scaled_dot_product_attention, now attention_mask is of boolean type
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False
        )

        return attn_output
    
    def after_total_layers(self):
        pass    



class KVSharingMaskGuidedAdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        text_length,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        start_step=2,
        start_layer=5,
        layer_idx=None,
        step_idx=None,
        overlap_size=6,
        sampling_num_frames=13,
        end_step=50,       
        end_layer=30,       
        mask_save_dir=None,
        ref_token_idx=None,
        cur_token_idx=None,
        attn_map_step_idx=None,
        attn_map_layer_idx=None,
        thres=0.1,
        num_prompts=None,
        num_transition_blocks=None,
        longer_mid_segment=None,
        is_edit=False,
    ):
        super().__init__()
        print("KVSharingMaskGuidedAdaLNMixin Init")
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.text_length = text_length
        
        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )

        self.cur_step = 0
        self.cur_layer = 0
        self.num_prompts = num_prompts
        self.num_transition_blocks = num_transition_blocks
        self.longer_mid_segment = longer_mid_segment
        self.count_segment = 0
        self.is_edit = is_edit
        
        self.end_step = end_step
        self.end_layer = end_layer 
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))

        self.attn_map_step_idx = attn_map_step_idx if attn_map_step_idx is not None else [5,10,15,20,25,30,35,40,45,50]
        self.attn_map_layer_idx = attn_map_layer_idx if attn_map_layer_idx is not None else [5,10,15,20,25,30]

        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx
        self.thres = thres

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)
            self.attn_controller = AttentionMapController(self.mask_save_dir, self.text_length, self.height, self.width, self.compressed_num_frames, 
                                                      thres=self.thres)
            
    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)
        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)
        
        # text_attention_input[-1] = text_attention_input[-1] + 0.2

        attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)

        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        # text_gate_msa = text_gate_msa + 4
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)

            
        self.cur_layer += 1
        if self.cur_layer == self.num_layers:
            self.cur_layer = 0
            if self.is_edit:  # ever step, we just use two segments
                self.cur_step += 1
            else:  # every step, we use very long segments
                self.count_segment += 1
                # if self.count_segment == 2 * (self.num_prompts - 1) :
                total_segments = self.num_prompts + self.num_transition_blocks * (self.num_prompts - 1) + self.longer_mid_segment * (self.num_prompts - 2)
                # Every calculation, we use two segments, and stride=1. when equals, we reachs the last segment combination of one step
                if self.count_segment == total_segments - 1: 
                    self.count_segment = 0
                    self.cur_step += 1
            self.after_total_layers()
            
        return hidden_states

    def reinit(self, parent_model=None):
        self.cur_step = 0
        self.cur_layer = 0
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict 
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        self.attn_controller.save_cur_attn_map(query_layer, key_layer, cur_step=self.cur_step, layer_id=self.cur_layer)
        text_length = kwargs["text_length"]
        total_length = query_layer.shape[-2]

        H = self.height
        W = self.width
        if self.cur_step in self.step_idx and self.cur_layer in self.layer_idx:
            qu_s, qu_t, qc_s, qc_t = query_layer.chunk(4)
            ku_s, ku_t, kc_s, kc_t = key_layer.chunk(4)
            vu_s, vu_t, vc_s, vc_t = value_layer.chunk(4)

            # source branch
            out_u_s = old_impl(qu_s, ku_s, vu_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            out_c_s = old_impl(qc_s, kc_s, vc_s, attention_mask, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)

            if self.attn_controller.get_cross_attn_sum() is None or self.attn_controller.get_cross_attn_count() == 0:
                self.attn_controller.set_self_attn_mask(None)
                out_u_t = self.attn_batch(qu_t, ku_s, vu_s, None, self.ref_token_idx, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
                out_c_t = self.attn_batch(qc_t, kc_s, vc_s, None, self.ref_token_idx,attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            
            else:
                mask = self.attn_controller.aggregate_cross_attn_map(token_idx=self.ref_token_idx)  # (4, F, H, W)
                mask_source = mask[-2]  # (F, H, W)
                self_attns_mask = mask_source.flatten()
                self.attn_controller.set_self_attn_mask(self_attns_mask)
                if self.mask_save_dir is not None:
                    concat_mask_source = concatenate_frames(mask_source)
                    filename = f"mask_s_step{self.cur_step}_layer{self.cur_layer}_all_frames.png"
                    base_path = os.path.join(self.mask_save_dir, f"segment{self.count_segment}")
                    os.makedirs(base_path, exist_ok=True)
                    save_path = os.path.join(base_path, filename)
                    save_image(concat_mask_source, save_path)
                    
                out_u_t = self.attn_batch(qu_t, ku_s, vu_s, None, self.ref_token_idx, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
                out_c_t = self.attn_batch(qc_t, kc_s, vc_s, None, self.ref_token_idx, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs)
            
            if self.attn_controller.get_self_attn_mask() is not None:
                mask = self.attn_controller.aggregate_cross_attn_map(token_idx=self.cur_token_idx)  # (4, F, H, W)
                mask_target = mask[-1]  # (F, H, W)
                spatial_mask = mask_target.reshape(-1, 1)
                if self.mask_save_dir is not None:
                    concat_mask_target = concatenate_frames(mask_target)
                    filename = f"mask_t_step{self.cur_step}_layer{self.cur_layer}_all_frames.png"
                    base_path = os.path.join(self.mask_save_dir, f"segment{self.count_segment}")
                    os.makedirs(base_path, exist_ok=True)
                    save_path = os.path.join(base_path, filename)
                    save_image(concat_mask_target, save_path)
                # binarize the mask
                thres = self.thres
                spatial_mask[spatial_mask >= thres] = 1
                spatial_mask[spatial_mask < thres] = 0
                
                out_u_t_fg, out_u_t_bg = out_u_t.chunk(2)
                out_c_t_fg, out_c_t_bg = out_c_t.chunk(2)
                
                # Adjust the shape of spatial_mask to match out_u_t_fg and out_u_t_bg
                B_fg, H, L, D = out_u_t_fg.shape
                text_length = self.text_length
                video_length = L - text_length

                # # Create a complete mask, with the text part all set to 1, and the video part using spatial_mask
                # full_mask = torch.ones((B_fg, H, L, 1), device=out_u_t_fg.device, dtype=query_layer.dtype)
                # full_mask[:, :, text_length:, :] = spatial_mask.view(1, 1, video_length, 1).expand(B_fg, H, -1, -1)
                
                # Create the mask for the text part
                text_mask = torch.zeros((B_fg, H, text_length, 1), device=out_u_t_fg.device, dtype=query_layer.dtype)
                # Set the tokens related to the foreground to 1, and others to 0
                text_mask[:, :, self.cur_token_idx, :] = 1
                # Create the mask for the video part
                video_mask = spatial_mask.view(1, 1, video_length, 1).expand(B_fg, H, -1, -1)
                # Concatenate text_mask and video_mask to form the complete mask
                full_mask = torch.cat([text_mask, video_mask], dim=2)
                
                out_u_t = out_u_t_fg * full_mask + out_u_t_bg * (1 - full_mask)
                out_c_t = out_c_t_fg * full_mask + out_c_t_bg * (1 - full_mask)
                
                # Set self self-attention mask to None
                self.attn_controller.set_self_attn_mask(None)

            out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)
            return out    
            
        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )

    def attn_batch(self, q, k, v, attention_mask, ref_token_idx, attention_dropout, log_attention_weights, scaling_attention_score, **kwargs):
        # Ensure the input tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        self_attns_mask = self.attn_controller.get_self_attn_mask()
        if self_attns_mask is not None:
            # Binarize self_attns_mask
            thres = self.thres
            self_attns_mask = (self_attns_mask >= thres).float()
        
            # Get text length and total length
            text_length = self.text_length
            total_length = q.shape[-2]  # Use the shape of q instead of attention_mask
            video_length = total_length - text_length
            
            # Create a complete mask, including text and video parts
            full_mask = torch.zeros((total_length, total_length), dtype=q.dtype, device=q.device)
            
            # Expand self_attns_mask to 2D and place it in the video part of full_mask
            self_attns_mask_2d = self_attns_mask.unsqueeze(0).expand(video_length, video_length)
            full_mask[text_length:, text_length:] = self_attns_mask_2d
            
            # Create attention masks for foreground and background, starting from zero instead of cloning
            min_value = torch.finfo(q.dtype).min
            fg_mask = torch.zeros((total_length, total_length), dtype=q.dtype, device=q.device)
            bg_mask = torch.zeros((total_length, total_length), dtype=q.dtype, device=q.device)
            
            # import pdb; pdb.set_trace()
            # Set the text part to 0 (allowing attention)
            fg_mask[:text_length, :] = 0
            fg_mask[:, :text_length] = 0
            bg_mask[:text_length, :] = 0
            bg_mask[:, :text_length] = 0
            
            # Set the video part based on full_mask
            fg_mask[text_length:, text_length:] = torch.where(
                full_mask[text_length:, text_length:] == 1,
                torch.zeros_like(fg_mask[text_length:, text_length:]),
                min_value
            )
            bg_mask[text_length:, text_length:] = torch.where(
                full_mask[text_length:, text_length:] == 0,
                torch.zeros_like(bg_mask[text_length:, text_length:]),
                min_value
            )
            
            # Calculate attention for foreground and background
            dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
            
            attn_output_fg = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=fg_mask,
                dropout_p=dropout_p,
                is_causal=False
            )
            
            attn_output_bg = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=bg_mask,
                dropout_p=dropout_p,
                is_causal=False
            )
            
            attn_output = torch.cat([attn_output_fg, attn_output_bg], dim=0)
        else:
            dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,  
                dropout_p=dropout_p,
                is_causal=False
            )

        return attn_output

    def after_total_layers(self):
        self.attn_controller.reset_cross_attns() 

str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class DiffusionTransformer(BaseModel):
    def __init__(
        self,
        transformer_args,
        num_frames,
        time_compressed_rate,
        latent_width,
        latent_height,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_layers,
        num_attention_heads,
        text_length,
        elementwise_affine,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time="adaln",
        adm_in_channels=None,
        parallel_output=True,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        use_SwiGLU=False,
        use_RMSNorm=False,
        zero_init_y_embed=False,
        **kwargs,
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.text_length = text_length
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        self.inner_hidden_size = hidden_size * 4
        self.zero_init_y_embed = zero_init_y_embed
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32
        
        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            kwargs["layernorm"] = partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6)

        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        module_configs = modules
        self._build_modules(module_configs)

        if use_SwiGLU:
            self.add_mixin(
                "swiglu", SwiGLUMixin(num_layers, hidden_size, self.inner_hidden_size, bias=False), reinit=True
            )

    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
                if self.zero_init_y_embed:
                    nn.init.constant_(self.label_emb[0][2].weight, 0)
                    nn.init.constant_(self.label_emb[0][2].bias, 0)
            else:
                raise ValueError()

        pos_embed_config = module_configs["pos_embed_config"]
        self.add_mixin(
            "pos_embed",
            instantiate_from_config(
                pos_embed_config,
                height=self.latent_height // self.patch_size,
                width=self.latent_width // self.patch_size,
                compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                hidden_size=self.hidden_size,
            ),
            reinit=True,
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.add_mixin(
            "patch_embed",
            instantiate_from_config(
                patch_embed_config,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                in_channels=self.in_channels,
            ),
            reinit=True,
        )
        if self.input_time == "adaln":
            self.adaln_layer_config = module_configs["adaln_layer_config"]
            self.add_mixin(
                "adaln_layer",
                instantiate_from_config(
                    self.adaln_layer_config,
                    height=self.latent_height // self.patch_size,
                    width=self.latent_width // self.patch_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                    hidden_size_head=self.hidden_size // self.num_attention_heads,
                    time_embed_dim=self.time_embed_dim,
                    elementwise_affine=self.elementwise_affine,
                    text_length=self.text_length,
                ),
            )
        else:
            raise NotImplementedError

        final_layer_config = module_configs["final_layer_config"]
        self.add_mixin(
            "final_layer",
            instantiate_from_config(
                final_layer_config,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                time_embed_dim=self.time_embed_dim,
                latent_width=self.latent_width,
                latent_height=self.latent_height,
                elementwise_affine=self.elementwise_affine,
            ),
            reinit=True,
        )

        if "lora_config" in module_configs:
            lora_config = module_configs["lora_config"]
            self.add_mixin("lora", instantiate_from_config(lora_config, layer_num=self.num_layers), reinit=True)

        return

    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs["seq_length"] = t * h * w // (self.patch_size**2)
        kwargs["images"] = x
        kwargs["emb"] = emb
        kwargs["encoder_outputs"] = context
        kwargs["text_length"] = context.shape[1]

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
        output = super().forward(**kwargs)[0]

        return output

    def switch_adaln_layer(self, mixin_class_name):
        """
        Switch the 'adaln_layer' module to the specified mixin class.
        
        Args:
            mixin_class_name (str): The name of the mixin class to switch to.
        """
        # Remove the existing 'adaln_layer' mixin
        self.del_mixin('adaln_layer')

        # Update the target in the adaln_layer_config
        self.adaln_layer_config["target"] = f"dit_video_concat.{mixin_class_name}"

        # Instantiate the new mixin with the updated config
        new_mixin = instantiate_from_config(
            self.adaln_layer_config,
            height=self.latent_height // self.patch_size,
            width=self.latent_width // self.patch_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
            hidden_size_head=self.hidden_size // self.num_attention_heads,
            time_embed_dim=self.time_embed_dim,
            elementwise_affine=self.elementwise_affine,
            text_length=self.text_length,
        )

        # Ensure the new mixin has the same dtype as the model
        new_mixin = new_mixin.to(self.dtype)

        # Add the new mixin
        self.add_mixin('adaln_layer', new_mixin)
        
        print(f"Switch to {mixin_class_name}")
        
        

        
