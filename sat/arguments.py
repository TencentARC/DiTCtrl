import argparse
import os
import torch
import json
import warnings
import omegaconf
from omegaconf import OmegaConf
from sat.helpers import print_rank0
from sat import mpu
from sat.arguments import set_random_seed
from sat.arguments import add_training_args, add_evaluation_args, add_data_args
import torch.distributed

import yaml
import re


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")
    group.add_argument("--base", type=str, nargs="*", help="config for input and saving")
    group.add_argument("--custom-config", type=str, default="", help="custom config for custom case")
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )
    group.add_argument("--force-pretrain", action="store_true")
    group.add_argument("--device", type=int, default=-1)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--log-image", type=bool, default=True)
    group.add_argument("--start-step", type=int, default=4, help="Starting step for masactrl")
    group.add_argument("--start-layer", type=int, default=26, help="Starting layer for masactrl")
    group.add_argument("--layer-idx", type=int, default=None, help="Layer index for masactrl")
    group.add_argument("--step-idx", type=int, default=None, help="Step index for masactrl")
    group.add_argument("--end-step", type=int, default=50, help="Ending step for masactrl")
    group.add_argument("--end-layer", type=int, default=30, help="Ending layer for masactrl")
    group.add_argument("--mask-save-dir", type=str, default="", help="Directory to save masks")
    group.add_argument("--attn-map-step-idx", nargs="+", type=int, default=[4,14,24,34,44,49], help="Attention map step indices")
    group.add_argument("--attn-map-layer-idx", nargs="+", type=int, default=[4,14,24,29], help="Attention map layer indices")
    group.add_argument("--thres", type=float, default=0.1, help="Threshold for masactrl")
    group.add_argument("--prompts", type=str, help="Multiple prompts separated by semicolons")
    group.add_argument("--ref-token-idx", type=int, nargs="+", help="Reference token indices")
    group.add_argument("--cur-token-idx", type=int, nargs="+", help="Current token indices")
    group.add_argument("--reweight-token-idx", type=int, nargs="+",help="Reweight token indices")
    group.add_argument("--reweight-scale", type=float, help="Reweight scale")
    group.add_argument("--is-run-isolated", type=bool, default=False, help="If running isolated video for comparison")
    group.add_argument("--single-prompt-length", type=int, default=0, help="Length of single prompt")

    return parser


def add_sampling_config_args(parser):
    """Sampling configurations"""

    group = parser.add_argument_group("sampling", "Sampling Configurations")
    group.add_argument("--output-dir", type=str, default="samples")
    group.add_argument("--input-dir", type=str, default=None)
    group.add_argument("--input-type", type=str, default="cli")
    group.add_argument("--input-file", type=str, default="input.txt")
    group.add_argument("--final-size", type=int, default=2048)
    group.add_argument("--sdedit", action="store_true")
    group.add_argument("--grid-num-rows", type=int, default=1)
    group.add_argument("--force-inference", action="store_true")
    group.add_argument("--lcm_steps", type=int, default=None)
    group.add_argument("--sampling-num-frames", type=int, default=32)
    group.add_argument("--sampling-fps", type=int, default=8)
    group.add_argument("--only-save-latents", type=bool, default=False)
    group.add_argument("--only-log-video-latents", type=bool, default=False)
    group.add_argument("--latent-channels", type=int, default=32)
    group.add_argument("--image2video", action="store_true")
    # Add AdaLNMixin_NAMES argument
    group.add_argument("--adaln-mixin-names", nargs="+", default=None, help="Names of AdaLNMixin modules to use.")
    group.add_argument('--num_transition_blocks', type=int, default=2,
                      help='Number of transition blocks between each pair of prompts')
    # for long video generation
    group.add_argument("--overlap-size", type=int, default=4)
    group.add_argument('--longer_mid_segment', type=int, default=1,
                      help='Additional segments for middle prompts')
    group.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')
    group.add_argument('--is-edit', type=bool, default=False, help='Whether to run edit mode')
    return parser


def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description="sat")
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_model_config_args(parser)
    parser = add_sampling_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)

    import deepspeed

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args_list)
    args = process_config_to_args(args)

    if not args.train_data:
        print_rank0("No training data specified", level="WARNING")

    assert (args.train_iters is None) or (args.epochs is None), "only one of train_iters and epochs should be set."
    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000  # default 10k iters
        print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun

    if args.device == -1:
        if torch.cuda.device_count() == 0:
            args.device = "cpu"
        elif args.local_rank is not None:
            args.device = args.local_rank
        else:
            args.device = args.rank % torch.cuda.device_count()

    if args.local_rank != args.device and args.mode != "inference":
        raise ValueError(
            "LOCAL_RANK (default 0) and args.device inconsistent. "
            "This can only happens in inference mode. "
            "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
        )

    if args.rank == 0:
        print_rank0("using world size: {}".format(args.world_size))

    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data)

    if args.mode != "inference":  # training with deepspeed
        args.deepspeed = True
        if args.deepspeed_config is None:  # not specified
            deepspeed_config_path = os.path.join(
                os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
            )
            with open(deepspeed_config_path) as file:
                args.deepspeed_config = json.load(file)
            override_deepspeed_config = True
        else:
            override_deepspeed_config = False

    assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."

    if args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank0("Automatically set fp16=True to use ZeRO.")
        args.fp16 = True
        args.bf16 = False

    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        if override_deepspeed_config:  # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config

    # initialize distributed and random seed because it always seems to be necessary.
    initialize_distributed(args)
    args.seed = args.seed + mpu.get_data_parallel_rank()
    set_random_seed(args.seed)
    return args


def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == "cpu":
        pass
    else:
        torch.cuda.set_device(args.device)
    # Call the init process
    init_method = "tcp://"
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    init_method += args.master_ip + ":" + args.master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Set vae context parallel group equal to model parallel group
    from sgm.util import set_context_parallel_group, initialize_context_parallel

    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed:
        import deepspeed

        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # It seems that it has no negative influence to configure it even without using checkpointing.
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout.
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )

            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0

            print_rank0(str(e), level="DEBUG")

    return True


def generate_output_path(args):
    base_path = args.output_dir

    os.makedirs(base_path, exist_ok=True)
    
    # Create yaml file path
    yaml_path = os.path.join(base_path, "params.yaml")
    
    # Save only the key parameters in horse.yaml
    save_keys = [
        'latent_channels', 'mode', 'load', 'batch_size', 
        'sampling_num_frames', 'sampling_fps', 'fp16',
        'force_inference', 'is_run_isolated', 'seed',
        'output_dir', 'adaln_mixin_names',
        'start_step', 'end_step', 'start_layer', 'end_layer',
        'layer_idx', 'step_idx', 'thres',
        'attn_map_step_idx', 'attn_map_layer_idx', 'mask_save_dir',
        'overlap_size', 'num_transition_blocks', 'longer_mid_segment',
        'ref_token_idx', 'cur_token_idx', 'prompts',
        'single_prompt_length', 'is_edit', 'reweight_token_idx', 'reweight_scale'
    ]
    
    clean_args = {k: getattr(args, k) for k in save_keys if hasattr(args, k)}
    # If single_prompt_length > 0, only save the first prompt
    if hasattr(args, 'single_prompt_length') and args.single_prompt_length > 0:
        if isinstance(clean_args.get('prompts', []), list) and len(clean_args['prompts']) > 0:
            clean_args['prompts'] = [clean_args['prompts'][0]] 
    # Create a configuration with the same format as horse.yaml
    yaml_config = {
        'args': clean_args
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    # Update mask save path
    args.mask_save_dir = os.path.join(base_path, "attn_map")
    
    return base_path

def process_config_to_args(args):
    """Fetch args from only --base"""
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    # Load custom config if it exists
    if args.custom_config and os.path.exists(args.custom_config):
        custom_config = OmegaConf.load(args.custom_config)
        # Merge custom config into base config, custom config values will override base config values
        config = OmegaConf.merge(config, custom_config)
    args_config = config.pop("args", OmegaConf.create())
    for key in args_config:
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            arg = OmegaConf.to_object(args_config[key])
        else:
            arg = args_config[key]
        if hasattr(args, key):
            setattr(args, key, arg)
    # process single_prompt_length
    if hasattr(args, 'single_prompt_length') and args.single_prompt_length > 0:
        if args.prompts and len(args.prompts) == 1:
            original_prompt = args.prompts[0]
            args.prompts = [original_prompt] * args.single_prompt_length
            print_rank0(f"Extended single prompt to {args.single_prompt_length} copies")  
        else:
            raise ValueError("Single prompt length is set, but no single prompt is provided.")
            
    # Convert the semicolon-separated prompts string to a list
    args.output_dir = generate_output_path(args)
    args.mask_save_dir = os.path.join(args.output_dir, "attn_map")
    args.num_prompts = len(args.prompts)
    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())
        args.model_config = model_config
        
        adaln_params = args.model_config.network_config.params.modules.adaln_layer_config.params
        # Register new parameters to adaln_params
        params_to_register = [
            'overlap_size', 'sampling_num_frames',
            'start_step', 'start_layer', 
            'layer_idx', 'step_idx', 'end_step', 'end_layer',
            'mask_save_dir', 'ref_token_idx', 'cur_token_idx', 
            'attn_map_step_idx', 'attn_map_layer_idx', 'thres',
            'num_prompts', 'num_transition_blocks', 'longer_mid_segment',
            'is_edit', 'reweight_token_idx', 'reweight_scale'
        ]

        for param in params_to_register:
            if hasattr(args, param):
                adaln_params[param] = getattr(args, param)

    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())
        args.data_config = data_config

    return args
