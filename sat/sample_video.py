import os
import math
import argparse, random
import yaml
import textwrap
import warnings 
import sys 
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.arguments import set_random_seed
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from sat.arguments import set_random_seed

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
    except EOFError as e:
        pass


def read_from_txt_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt

def read_from_yaml_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        data = yaml.safe_load(fin)
        
    prompts = data.get('prompts', [])
    cnt = -1
    
    for item in prompts:
        cnt += 1
        if cnt % world_size != rank:
            continue
        yield item, cnt

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = value_dict["prompt"]
            batch_uc["txt"] = value_dict["negative_prompt"]
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    total_frames = sum(vid.shape[0] for vid in video_batch)
    save_path = f"{save_path}_frames{total_frames}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames = []
    for vid in video_batch:
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            frames.append(frame)
    
    with imageio.get_writer(save_path, fps=fps, quality=8, format='FFMPEG') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video has been saved to: {save_path}")


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr
def print_current_gpu_memory(device):
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

def process_multi_prompt_video_with_adaln(model, args, c_total, uc_total, 
                       prompts, cnt, adaln_name, device, 
                       sample_func,
                       tile_size, 
                       overlap_size,
                       long_video_size, 
                       C, H, W, F, 
                       randn_noise):  
    set_random_seed(args.seed)
    model.switch_adaln_layer(adaln_name)
    load_checkpoint(model, args)
    model.to(device)
    model.eval()
    
    samples_z = sample_func(
        c_total,
        uc=uc_total,
        randn=randn_noise,
        tile_size = tile_size,
        overlap_size = overlap_size,
    )
    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

    print_current_gpu_memory(device)
    # Unload the model from GPU to save GPU memory
    model.to("cpu")
    torch.cuda.empty_cache()
    first_stage_model = model.first_stage_model
    first_stage_model = first_stage_model.to(device)

    latent = 1.0 / model.scale_factor * samples_z

    # Decode latent serial to save GPU memory
    recons = []
    loop_num = (long_video_size - 1) // 2
    for i in range(loop_num):
        if i == 0:
            start_frame, end_frame = 0, 3
        else:
            start_frame, end_frame = i * 2 + 1, i * 2 + 3 
        if i == loop_num - 1:
            clear_fake_cp_cache = True
        else:
            clear_fake_cp_cache = False
        with torch.no_grad():
            recon = first_stage_model.decode(
                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
            )
        recons.append(recon)
    
    recon = torch.cat(recons, dim=2).to(torch.float32)
    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

    save_path = os.path.join(
            args.output_dir,
            "MultiPrompt",
        )
    if mpu.get_model_parallel_rank() == 0:
        save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)
    
    del samples_z, latent, recons, recon, samples_x, samples
    torch.cuda.empty_cache()
    print(f"Finished MultiPrompt_{adaln_name}")
    print_current_gpu_memory(device)


def interpolate_conditions(c1, c2, weight_1, weight_2):
    """
    Linear interpolation of two conditions
    Args:
        c1: The first condition
        c2: The second condition
        weight_1: The weight of the first condition
        weight_2: The weight of the second condition
    Returns:
        c_interpolated: The interpolated condition
    """
    # Directly interpolate the tensor without handling complex dictionary structures
    return weight_1 * c1 + weight_2 * c2

def calculate_segments_per_prompt(prompts_length, num_transition_blocks, longer_mid_segment):
    """
    Calculate how many base segments each prompt should have
    Args:
        prompts_length: The number of prompts
        num_transition_blocks: The number of transition blocks between each two prompts
        longer_mid_segment: The additional segments for the middle prompt
    """
    if prompts_length <= 2:
        return [1] * prompts_length
    
    segments = []
    for i in range(prompts_length):
        if i == 0 or i == prompts_length - 1:
            # The first and last prompts use 1 base segment
            segments.append(1)
        else:
            # The middle prompts increase by longer_mid_segment segments
            segments.append(1 + longer_mid_segment)
    return segments

def calculate_total_segments(prompts_length, num_transition_blocks, longer_mid_segment):
    """Calculate the total number of segments, including transition blocks and additional middle segments"""
    if prompts_length <= 1:
        return prompts_length
        
    # Calculate the number of base segments
    base_segments = prompts_length
    # Calculate the number of additional segments for middle prompts
    if prompts_length > 2:
        extra_mid_segments = (prompts_length - 2) * longer_mid_segment
    else:
        extra_mid_segments = 0
    # Calculate the number of transition blocks
    transition_segments = (prompts_length - 1) * num_transition_blocks
    
    return base_segments + extra_mid_segments + transition_segments

def generate_conditioning_parts(prompts, model, num_samples, num_transition_blocks, longer_mid_segment):
    """
    Generate conditions, supporting transition blocks and longer mid segments
    """
    c_total = []
    uc_total = []
    
    # Calculate the base segment count for each prompt
    segments_per_prompt = calculate_segments_per_prompt(len(prompts), num_transition_blocks, longer_mid_segment)
    
    # Generate base conditions for each prompt
    base_conditions = []
    base_uc_conditions = []
    for prompt in prompts:
        current_batch = {'txt': prompt}
        current_batch_uc = {'txt': ""}
        
        c, uc = model.conditioner.get_unconditional_conditioning(
            current_batch,
            batch_uc=current_batch_uc,
            force_uc_zero_embeddings=["txt"],
        )
        for k in c:
            if k != "crossattn":
                c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
        base_conditions.append(c)
        base_uc_conditions.append(uc)
    
    # Generate the final condition sequence
    for i in range(len(prompts)):
        # Add the base segments for the current prompt
        for _ in range(segments_per_prompt[i]):
            c_total.append(base_conditions[i])
            uc_total.append(base_uc_conditions[i])
        
        # If not the last prompt, add transition blocks
        if i < len(prompts) - 1:
            for j in range(num_transition_blocks):
                weight_2 = (j + 1) / (num_transition_blocks + 1)
                weight_1 = 1 - weight_2
                
                c_transition = {}
                for k in base_conditions[i].keys():
                    c_transition[k] = interpolate_conditions(
                        base_conditions[i][k],
                        base_conditions[i + 1][k],
                        weight_1,
                        weight_2
                    )
                c_total.append(c_transition)
                uc_total.append(base_uc_conditions[i] if weight_1 > weight_2 else base_uc_conditions[i + 1])

    return c_total, uc_total

def calculate_video_length(prompts_length, tile_size, overlap_size, num_transition_blocks, longer_mid_segment):
    """Calculate the total video length"""
    total_segments = calculate_total_segments(prompts_length, num_transition_blocks, longer_mid_segment)
    return tile_size + (total_segments - 1) * (tile_size - overlap_size)

def process_noise_blocks(randn_noise_original, prompts_length, tile_size, overlap_size, long_video_size, num_transition_blocks):
    """
    Process noise blocks, supporting a configurable number of transition blocks
    Args:
        num_transition_blocks: The number of transition blocks between each two prompts
    """
    # Calculate the total number of segments
    total_segments = prompts_length + (prompts_length - 1) * num_transition_blocks
    
    # Calculate the starting positions of all segments
    tile_starts = []
    current_start = 0
    while current_start + tile_size <= long_video_size:
        tile_starts.append(current_start)
        current_start += tile_size - overlap_size

    tile_indices = [list(range(start, start + tile_size)) for start in tile_starts]
    
    # Initialize randn_noise
    randn_noise = randn_noise_original.clone()
    
    # Process the noise for each segment
    for i in range(1, len(tile_indices)):
        prev_segment_indices = tile_indices[i - 1]
        current_segment_indices = tile_indices[i]

        # Calculate the actual overlap length
        overlap_len = min(overlap_size, len(current_segment_indices), len(prev_segment_indices))

        # Ensure the overlapping parts are the same
        overlap_indices_current = current_segment_indices[:overlap_len]
        overlap_indices_prev = prev_segment_indices[-overlap_len:]
        randn_noise[:, overlap_indices_current, :, :, :] = randn_noise[:, overlap_indices_prev, :, :, :]

    return randn_noise, tile_indices

def get_base_prompt_indices_with_longer_mid(tile_indices, prompts_length, num_transition_blocks, longer_mid_segment):
    """
    Get the indices corresponding to each base prompt (considering longer_mid_segment)
    Args:
        tile_indices: List of indices for all segments
        prompts_length: Number of prompts
        num_transition_blocks: Number of transition blocks between each two prompts
        longer_mid_segment: Additional segments for the middle prompt
    Returns:
        base_indices: List of base indices for each prompt
    """
    base_indices = []
    current_idx = 0
    
    for i in range(prompts_length):
        # Get the indices of the first segment of the current prompt
        base_indices.append(tile_indices[current_idx])
        
        # Calculate the starting position of the next prompt
        if i < prompts_length - 1:
            # Check if the current prompt is a middle prompt
            if i > 0 and i < prompts_length - 1:
                current_idx += (1 + longer_mid_segment)  # Base segment + additional segments
            else:
                current_idx += 1  # Only base segment
            current_idx += num_transition_blocks  # Add transition blocks
    
    return base_indices

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
        
    AdaLNMixin_NAMES = args.adaln_mixin_names
        
    load_checkpoint(model, args)
    model.eval()

    rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    print("rank and world_size", rank, world_size)

    image_size = [480, 720]

    sample_func_single = model.sample_single
    sample_func_multi_prompt = model.sample_multi_prompt
    H, W, C, F = image_size[0], image_size[1], args.latent_channels, 8
    tile_size = args.sampling_num_frames
    overlap_size = args.overlap_size
    num_transition_blocks = args.num_transition_blocks
    longer_mid_segment = args.longer_mid_segment
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device

    with torch.no_grad():
        # for prompts, cnt in tqdm(data_iter):
        if True:
            prompts = args.prompts
            cnt = 0
            # reload model on GPU
            model.to(device)
            set_random_seed(args.seed)
            print("\n")
            print("rank:", rank, "start to process", prompts, cnt)
            print_current_gpu_memory(device)
            
            long_video_size = calculate_video_length(
                len(prompts), 
                tile_size, 
                overlap_size, 
                num_transition_blocks,
                longer_mid_segment
            )
            shape=(long_video_size, C, H // F, W // F)
            batch_size = 1

            randn_noise_original = torch.randn(batch_size, *shape).to(torch.float32).to(device)
            randn_noise, tile_indices = process_noise_blocks(
                randn_noise_original, 
                len(prompts),
                tile_size, 
                overlap_size, 
                long_video_size,
                num_transition_blocks
            )
            print(f"Processing {len(prompts)} prompts: {prompts}")
            print(f"num of tile: {len(tile_indices)}")
            print(f"tile_indices: {tile_indices}")

            c_total, uc_total = generate_conditioning_parts(
                prompts, 
                model, 
                num_samples,
                num_transition_blocks,
                longer_mid_segment
            )

            ''''for comparison, single-prompt based video generation'''
            if args.is_run_isolated:
                iso_samples = []
                for index, prompt in enumerate(prompts): 
                    # reload model on GPU
                    model.to(device)
                    
                    # Get the indices of all prompts, considering longer_mid_segment
                    base_prompt_indices = get_base_prompt_indices_with_longer_mid(
                        tile_indices, 
                        len(prompts), 
                        args.num_transition_blocks,
                        longer_mid_segment
                    )
                    current_prompt_indices = base_prompt_indices[index]
                    
                    # calculate current prompt index 
                    condition_index = 0
                    for i in range(index):
                        if i == 0 or i == len(prompts) - 1:
                            condition_index += 1 
                        else:
                            condition_index += (1 + longer_mid_segment)
                        condition_index += args.num_transition_blocks
                    print(f"condition_index: {condition_index}")
                    print(f"isolated current_prompt_indices: {current_prompt_indices}")
                    samples_z = sample_func_single(
                        c_total[condition_index],
                        uc=uc_total[condition_index],
                        randn=randn_noise[:, current_prompt_indices]
                    )
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                    # Unload the model from GPU to save GPU memory
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    first_stage_model = model.first_stage_model
                    first_stage_model = first_stage_model.to(device)

                    latent = 1.0 / model.scale_factor * samples_z

                    # Decode latent serial to save GPU memory
                    recons = []
                    loop_num = (tile_size - 1) // 2
                    for i in range(loop_num):
                        if i == 0:
                            start_frame, end_frame = 0, 3
                        else:
                            start_frame, end_frame = i * 2 + 1, i * 2 + 3
                        if i == loop_num - 1:
                            clear_fake_cp_cache = True
                        else:
                            clear_fake_cp_cache = False
                        with torch.no_grad():
                            recon = first_stage_model.decode(
                                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                            )

                        recons.append(recon)

                    recon = torch.cat(recons, dim=2).to(torch.float32)
                    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                    save_path = os.path.join(
                        args.output_dir, 
                        "isolated",
                        str(index) + "_" + prompts[index].replace(" ", "_").replace("/", "")[:120]
                    )
                    if mpu.get_model_parallel_rank() == 0:
                        save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)
                    print(f'finish isolated_{index}')
                    iso_samples.append(samples)

                print_current_gpu_memory(device)
                iso_samples = torch.concat(iso_samples, dim=1)
                save_path = os.path.join(
                        args.output_dir, 
                        "isolated",
                        "all",
                    )
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(iso_samples, save_path, fps=args.sampling_fps)
                print(f'finish isolated all!!')
                
                del samples_z, latent, recons, recon, samples_x, samples
                torch.cuda.empty_cache()
            
            ''''multi-prompt based long video generation'''
            for i, adaln_name in enumerate(AdaLNMixin_NAMES):
                process_multi_prompt_video_with_adaln(model, args, 
                                                    c_total, uc_total,
                                                    prompts, cnt, adaln_name, device, 
                                                    sample_func_multi_prompt, #multi_prompt sample func
                                                    tile_size,
                                                    overlap_size,
                                                    long_video_size, 
                                                    C, H, W, F, 
                                                    randn_noise.clone())
            # For BaseVersion
            model.switch_adaln_layer('BaseAdaLNMixin')
            load_checkpoint(model, args)
            model.to(device)
            model.eval()

    
if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    
    sampling_main(args, model_cls=SATVideoDiffusionEngine)