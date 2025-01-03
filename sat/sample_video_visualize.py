import os
import math
import argparse, random
import yaml
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from sat.arguments import set_random_seed
from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import sys
import warnings
import textwrap


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
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
            batch["txt"] = np.repeat(value_dict["prompt"], repeats=math.prod(N) // len(value_dict["prompt"])).reshape(N).tolist()
            batch_uc["txt"] = np.repeat(value_dict["negative_prompt"], repeats=math.prod(N) // len(value_dict["negative_prompt"])).reshape(N).tolist()
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

    print(f"video saved to: {save_path}")


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


def process_with_adaln(model, args, c, uc, prompts, cnt, adaln_name, device, sample_func, T, C, H, W, F, randn_noise):
    set_random_seed(args.seed)
    model.switch_adaln_layer(adaln_name)
    load_checkpoint(model, args)
    model.to(device)
    model.eval()
    
    samples_z = sample_func(
        c,
        uc=uc,
        randn=randn_noise
    )
    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

    print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
    
    model.to("cpu")
    torch.cuda.empty_cache()
    first_stage_model = model.first_stage_model.to(device)

    latent = 1.0 / model.scale_factor * samples_z

    recons = []
    loop_num = (T - 1) // 2
    for i in range(loop_num):
        start_frame, end_frame = (0, 3) if i == 0 else (i * 2 + 1, i * 2 + 3)
        clear_fake_cp_cache = i == loop_num - 1
        with torch.no_grad():
            recon = first_stage_model.decode(
                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
            )
        recons.append(recon)

    recon = torch.cat(recons, dim=2).to(torch.float32)
    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

    # [B, T, C, H, W]
    B, T, C, H, W = samples.shape
    # [1, T, C, H, B*W]
    horizontally_stacked = torch.zeros(1, T, C, H, B*W)
    for i in range(B):
        horizontally_stacked[0, :, :, :, i*W:(i+1)*W] = samples[i]
    
    save_path = os.path.join(args.output_dir, f"visualization")
    print(f"Saving horizontally combined video to {save_path}")
    save_video_as_grid_and_mp4(horizontally_stacked, save_path, fps=args.sampling_fps)

    del samples_z, latent, recons, recon, samples_x, samples
    torch.cuda.empty_cache()
    print(f"Finished processing {len(prompts)} prompts with {adaln_name}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    
    AdaLNMixin_NAMES = args.adaln_mixin_names
    
    load_checkpoint(model, args)
    model.eval()

    image_size = [480, 720]
    
    sample_func = model.sample_single #sample_single
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
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
            print("start to process", prompts, cnt)
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
            num_samples = [len(prompts)]  
            # random noise
            shape=(T, C, H // F, W // F)
            randn_noise_one_batch = torch.randn(1, *shape).to(torch.float32).to(device)
            # expand to len(prompts)
            randn_noise = randn_noise_one_batch.expand(len(prompts), *shape)
            
            print(f"Processing {len(prompts)} prompts: {prompts}")
            value_dict = {
                "prompt": prompts,
                "negative_prompt": ["" for _ in prompts],
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(f"{key}: {batch[key].shape}")
                elif isinstance(batch[key], list):
                    print(f"{key}: {[len(l) for l in batch[key]]}")
                else:
                    print(f"{key}: {batch[key]}")
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            
            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            # reload model on GPU
            model.to(device)
        
            for i, adaln_name in enumerate(AdaLNMixin_NAMES):
                process_with_adaln(model, args, c, uc, prompts, cnt, adaln_name, device, sample_func, T, C, H, W, F, randn_noise.clone())
                
                next_adaln_name = AdaLNMixin_NAMES[(i + 1) % len(AdaLNMixin_NAMES)]
            
            model.switch_adaln_layer(next_adaln_name)
            load_checkpoint(model, args)
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