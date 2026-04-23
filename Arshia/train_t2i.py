import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from mmdit import MMDiT
from loss import SILoss

from dataset import MSCOCO256Features
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type, resolution=256):
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Skip ORCA params if not in EMA (ORCA is training-only, removed at inference)
        if name not in ema_params:
            continue
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    # DDP kwargs to handle unused parameters (when proj_coeff=0, projector params are unused)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != 'None':
        from utils import load_encoders
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
    else:
        encoders, encoder_types, architectures = [None], [None], [None]
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    #block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = MMDiT(
        input_size=latent_size,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        use_orca=args.use_orca,
        orca_rank=args.orca_rank,
        orca_ortho_method=args.orca_ortho_method,
        orca_normalize_delta=args.orca_normalize_delta,
        orca_stop_grad_gap=args.orca_stop_grad_gap,
        orca_normalize_gap=args.orca_normalize_gap,
        orca_clamp_gap=args.orca_clamp_gap,
        orca_ddl_style=args.orca_ddl_style,
        orca_v2=args.orca_v2,
        orca_loss_only=args.orca_loss_only,
        orca_monitor_only=args.orca_monitor_only,
        orca_loss_mode=args.orca_loss_mode,
        orca_new_formulation=args.orca_new_formulation,
        orca_normalize_projections=args.orca_normalize_projections,
        orca_gate_type=args.orca_gate_type,
        orca_gate_threshold=args.orca_gate_threshold,
        orca_eta=args.orca_eta,
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)

    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Missing keys (expected for new ORCA params): {missing[:5]}...")
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'], strict=False)
            print(f"Resumed from checkpoint: {args.resume}")
        else:
            # Handle old checkpoint format
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Missing keys (expected for new ORCA params): {missing[:5]}...")
            print(f"Resumed from checkpoint (old format): {args.resume}")
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    if args.use_orca and args.orca_embeddings_dir:
        from dataset_orca import MSCOCO256FeaturesORCA, collate_fn_orca
        if accelerator.is_main_process:
            logger.info(f"Using ORCA dataset with embeddings from {args.orca_embeddings_dir}")
        dataset_factory = MSCOCO256FeaturesORCA(
            repa_path=args.data_dir,
            orca_path=args.orca_embeddings_dir,
            cfg=True,
            p_uncond=args.cfg_prob,
            mode='train'
        )
        train_dataset = dataset_factory.train
        collate_fn = collate_fn_orca
    else:
        train_dataset = MSCOCO256Features(path=args.data_dir).train
        collate_fn = None

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    # Handle different dataset formats: ORCA returns dict, vanilla returns tuple
    if args.use_orca:
        batch = next(iter(train_dataloader))
        gt_xs = batch['z']
    else:
        _, gt_xs, _, _ = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    # Create sampling noise:
    xT = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dataloader:
            # Handle ORCA dataset (dict) vs standard dataset (tuple)
            if args.use_orca:
                raw_image = batch['x']
                x = batch['z']
                context = batch['c']
                clip_text_emb = batch['clip_text']
                t5_emb = batch['t5_text']
                dinov2_emb = batch['dinov2']
                raw_captions = None  # Not used with ORCA
            else:
                raw_image, x, context, raw_captions = batch
                clip_text_emb = None
                t5_emb = None
                dinov2_emb = None

            if global_step == 0:
                ys = context[:sample_batch_size].to(device) # handed-coded
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device) if not args.use_orca else x.to(device)
            context = context.to(device)

            # Move ORCA embeddings to device
            if args.use_orca:
                clip_text_emb = clip_text_emb.to(device)
                t5_emb = t5_emb.to(device)
                dinov2_emb = dinov2_emb.to(device)

            z = None
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                # Skip DINOv2 encoding if proj_coeff=0 (huge speedup for vanilla)
                if args.proj_coeff > 0:
                    zs = []
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(
                                raw_image, encoder_type, resolution=args.resolution
                                )
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z = z[:, 1:]
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)
                else:
                    zs = []

            with accelerator.accumulate(model):
                model_kwargs = dict(
                    context=context,
                    clip_text_emb=clip_text_emb,
                    t5_emb=t5_emb,
                    dinov2_emb=dinov2_emb,
                )

                # Get ORCA diagnostics every step for diagnostic run or every 100 steps normally
                # IMPORTANT: Always get diagnostics when using loss modes (alignment_loss is in diagnostics)
                needs_orca_loss = args.use_orca and (args.orca_loss_only or getattr(args, 'orca_loss_mode', 'none') != 'none')
                return_orca_diagnostics = args.use_orca and (args.diagnostic_mode or global_step % 100 == 0 or needs_orca_loss)

                if args.proj_coeff > 0:
                    result = loss_fn(model, x, model_kwargs, zs=zs, return_orca_diagnostics=return_orca_diagnostics)
                    if return_orca_diagnostics and len(result) == 3:
                        loss, proj_loss, orca_diagnostics = result
                    else:
                        loss, proj_loss = result
                        orca_diagnostics = None
                    loss_mean = loss.mean()
                    proj_loss_mean = proj_loss.mean()
                    loss = loss_mean + proj_loss_mean * args.proj_coeff
                else:
                    result = loss_fn(model, x, model_kwargs, zs=zs, return_orca_diagnostics=return_orca_diagnostics)
                    if return_orca_diagnostics and len(result) == 3:
                        loss, proj_loss, orca_diagnostics = result
                    else:
                        loss, proj_loss = result
                        orca_diagnostics = None
                    loss_mean = loss.mean()
                    proj_loss_mean = torch.tensor(0.0, device=loss.device)  # No proj loss when proj_coeff=0

                    # Add ORCA alignment loss if in loss-only mode or using loss_mode
                    if ((args.orca_loss_only or args.orca_loss_mode != "none") and
                        orca_diagnostics and "alignment_loss" in orca_diagnostics):
                        alignment_loss = orca_diagnostics["alignment_loss"]
                        loss = loss_mean + alignment_loss * args.orca_loss_coeff
                    else:
                        loss = loss_mean

                    # Temporary debug print for first 5 steps
                    if global_step < 5:
                        print(f"[DEBUG] Step {global_step}: orca_loss_only={args.orca_loss_only}, loss_mode={args.orca_loss_mode}, has_diagnostics={orca_diagnostics is not None}, has_alignment_loss={'alignment_loss' in orca_diagnostics if orca_diagnostics else False}")
                        if orca_diagnostics and "alignment_loss" in orca_diagnostics:
                            al = orca_diagnostics["alignment_loss"]
                            print(f"[DEBUG] alignment_loss={al}, requires_grad={al.requires_grad if hasattr(al, 'requires_grad') else 'N/A'}, loss_coeff={args.orca_loss_coeff}")

                ## optimization
                accelerator.backward(loss)

                # Compute ORCA gradient norms before optimizer step
                orca_grad_norms = {}
                if args.use_orca and (args.diagnostic_mode or global_step % 100 == 0) and accelerator.sync_gradients:
                    if hasattr(model, 'module'):
                        orca_module = model.module.orca_module if hasattr(model.module, 'orca_module') else None
                    else:
                        orca_module = model.orca_module if hasattr(model, 'orca_module') else None

                    if orca_module is not None:
                        # Compute gradient norms for ORCA parameters
                        with torch.no_grad():
                            # Log scale parameter gradient (single value)
                            if orca_module.log_scale.grad is not None:
                                orca_grad_norms['grad_log_scale'] = orca_module.log_scale.grad.abs().item()
                            else:
                                orca_grad_norms['grad_log_scale'] = 0.0

                            # Gate network gradient (handle MLP gate, cosine gate, and DDL-style)
                            if hasattr(orca_module, 'gate') and orca_module.gate is not None:
                                # MLP gate (has parameters)
                                gate_grads = []
                                for p in orca_module.gate.parameters():
                                    if p.grad is not None:
                                        gate_grads.append(p.grad.norm().item())
                                orca_grad_norms['grad_gate'] = sum(gate_grads) / len(gate_grads) if gate_grads else 0.0
                            elif hasattr(orca_module, 'beta_gate'):
                                # DDL-style beta gate
                                if orca_module.beta_gate.weight.grad is not None:
                                    orca_grad_norms['grad_gate'] = orca_module.beta_gate.weight.grad.norm().item()
                                else:
                                    orca_grad_norms['grad_gate'] = 0.0
                            else:
                                # Cosine gate or no gate (no learnable parameters)
                                orca_grad_norms['grad_gate'] = 0.0

                            # K mapping network last layer gradient
                            k_mapping_last_layer = orca_module.k_mapping.net[-1]
                            if k_mapping_last_layer.weight.grad is not None:
                                orca_grad_norms['grad_k_mapping'] = k_mapping_last_layer.weight.grad.norm().item()
                            else:
                                orca_grad_norms['grad_k_mapping'] = 0.0

                            # Projection layer gradients
                            if orca_module.proj_clip_text.weight.grad is not None:
                                orca_grad_norms['grad_proj_clip'] = orca_module.proj_clip_text.weight.grad.norm().item()
                            else:
                                orca_grad_norms['grad_proj_clip'] = 0.0

                            if orca_module.proj_t5.weight.grad is not None:
                                orca_grad_norms['grad_proj_t5'] = orca_module.proj_t5.weight.grad.norm().item()
                            else:
                                orca_grad_norms['grad_proj_t5'] = 0.0

                            if orca_module.proj_dinov2.weight.grad is not None:
                                orca_grad_norms['grad_proj_dinov2'] = orca_module.proj_dinov2.weight.grad.norm().item()
                            else:
                                orca_grad_norms['grad_proj_dinov2'] = 0.0

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers_t2i import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        y_null=torch.tensor(
                            train_dataset.empty_token
                            ).to(device).unsqueeze(0).repeat(ys.shape[0], 1, 1),
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }

            # Add ORCA diagnostics to logs every step in diagnostic mode or every 100 steps normally
            if args.use_orca and (args.diagnostic_mode or global_step % 100 == 0):
                if 'orca_diagnostics' in locals() and orca_diagnostics is not None:
                    # Add ORCA diagnostics with prefix
                    for key, value in orca_diagnostics.items():
                        logs[f"orca/{key}"] = value

                # Add ORCA gradient norms
                if orca_grad_norms:
                    for key, value in orca_grad_norms.items():
                        logs[f"orca/{key}"] = value

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # model
    parser.add_argument("--depth", type=int, default=24,
                        help="Number of MMDiT joint blocks. DiT-L=24, DiT-XL=28.")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Transformer hidden size. If unset, falls back to 32*depth (legacy). DiT-XL=1152.")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Number of attention heads. If unset, falls back to depth (legacy). DiT-XL=16.")
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/coco256_features")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    # ORCA arguments
    parser.add_argument("--use-orca", action="store_true", help="Enable ORCA T2I correction")
    parser.add_argument("--diagnostic-mode", action="store_true", help="Log ORCA diagnostics every step")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps for diagnostic runs")
    parser.add_argument("--orca-rank", type=int, default=4, help="ORCA K-matrix rank")
    parser.add_argument("--orca-ortho-method", type=str, default="qr", choices=["qr", "l2norm"],
                        help="Orthogonalization method for K matrix: 'qr' (default) or 'l2norm'")
    parser.add_argument("--orca-normalize-delta", action="store_true",
                        help="Normalize text projections before computing delta_y")
    parser.add_argument("--orca-stop-grad-gap", action="store_true",
                        help="Stop gradient on K when computing gap (halves K gradients)")
    parser.add_argument("--orca-normalize-gap", action="store_true",
                        help="Normalize gap vector to unit norm before reconstruction")
    parser.add_argument("--orca-clamp-gap", action="store_true",
                        help="Clamp gap norm to max 2.0 (preserves small gaps)")
    parser.add_argument("--orca-ddl-style", action="store_true",
                        help="Use DDL-style architecture with context-gated interpolation")
    parser.add_argument("--orca-v2", action="store_true",
                        help="Use ORCA v2 with cleaner architecture and geometric projections")
    parser.add_argument("--orca-new-formulation", action="store_true",
                        help="Use new ORCA formulation with delta_x-based gating")
    parser.add_argument("--orca-normalize-projections", action="store_true",
                        help="Normalize proj_dino and h_in_K to unit vectors in new formulation")
    parser.add_argument("--orca-gate-type", type=str, default="mlp", choices=["mlp", "cosine"],
                        help="Gate type for new formulation: mlp (learned) or cosine (geometric)")
    parser.add_argument("--orca-gate-threshold", type=float, default=0.0,
                        help="Hard threshold for gate values (zeros out gates below threshold)")
    parser.add_argument("--orca-eta", type=float, default=0.1,
                        help="Scale factor for ORCA corrections (default: 0.1)")
    parser.add_argument("--orca-loss-only", action="store_true",
                        help="ORCA computes alignment loss instead of modifying h_t")
    parser.add_argument("--orca-monitor-only", action="store_true",
                        help="ORCA only monitors metrics, no loss or h_t modification")
    parser.add_argument("--orca-loss-coeff", type=float, default=1.0,
                        help="Weight of ORCA alignment loss when using --orca-loss-only")
    parser.add_argument("--orca-loss-mode", type=str, default="none", choices=["none", "mse", "cosine"],
                        help="Type of alignment loss: none (correction), mse (gap loss), cosine (text-conditioned)")
    parser.add_argument("--orca-embeddings-dir", type=str, default=None,
                        help="Path to ORCA embeddings (clip_text, t5_text, dinov2)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()

    # Override max_train_steps if max_steps is provided
    if args.max_steps is not None:
        args.max_train_steps = args.max_steps
        print(f"Overriding max_train_steps to {args.max_steps} for diagnostic run")

    main(args)