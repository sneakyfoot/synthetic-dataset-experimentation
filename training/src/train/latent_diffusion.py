import argparse
import json
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from train.data import ConditionEncoder, ManifestDataset, build_image_transform, parse_condition_spec
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
    is_xformers_available,
)
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent-space DDPM on the custom manifest dataset (from scratch).")
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the dataset manifest JSONL file (beauty_all-TORCH-001-ML-15.json).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to filter from the manifest metadata (e.g. train or val).",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to the pretrained AutoencoderKL to encode/decode latents (trained from scratch).",
    )
    parser.add_argument(
        "--condition_spec",
        type=str,
        default="temperature:1,wind:3,wind_mag:1",
        help='Comma list of cond fields "name:dim". Use "none" for unconditional.',
    )
    parser.add_argument(
        "--cond_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping the condition vector (classifier-free style).",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Optional UNet2DModel config or checkpoint to initialize from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="latent-ddpm-model",
        help="The output directory where checkpoints and samples will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=256, help="Target image resolution before VAE encoding.")
    parser.add_argument(
        "--latent_channels_override",
        type=int,
        default=None,
        help="Override latent channels if different from VAE config (useful when custom VAE is trained with non-default channels).",
    )
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop after resizing.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly apply horizontal flip.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Number of images to generate for eval logging.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional dataset subset for debugging.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_images_epochs", type=int, default=5, help="How often to log generated samples.")
    parser.add_argument("--save_model_epochs", type=int, default=5, help="How often to save model weights.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help='Scheduler type: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", action="store_true", help="Whether to track EMA weights for the UNet.")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument(
        "--preserve_input_precision",
        action="store_true",
        help="Preserve 16/32-bit image precision instead of forcing an 8-bit RGB conversion.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision (fp16 or bf16).",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Experiment tracker to use.",
    )
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resume_from_checkpoint and args.overwrite_output_dir:
        logger.warning("Both resume_from_checkpoint and overwrite_output_dir set; resume will take precedence.")

    return args


def log_validation_images(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    args,
    weight_dtype: torch.dtype,
    vae_scale_factor: int,
    global_step: int,
    epoch: int,
    cond_dim: int,
    cross_attention_dim: int,
):
    unet = accelerator.unwrap_model(unet)
    device = accelerator.device
    vae = vae.to(device, dtype=weight_dtype)

    inference_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
    inference_scheduler.set_timesteps(args.ddpm_num_inference_steps)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_height = args.resolution // vae_scale_factor
    latent_width = args.resolution // vae_scale_factor
    latents = torch.randn(
        (
            args.eval_batch_size,
            unet.config.in_channels,
            latent_height,
            latent_width,
        ),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )

    encoder_hs = torch.zeros(
        (args.eval_batch_size, 1, cross_attention_dim),
        device=device,
        dtype=weight_dtype,
    )
    class_labels = None
    if cond_dim > 0:
        class_labels = torch.zeros((args.eval_batch_size, cond_dim), device=device, dtype=weight_dtype)

    with torch.no_grad():
        for t in inference_scheduler.timesteps:
            noise_pred = unet(latents, t, encoder_hidden_states=encoder_hs, class_labels=class_labels).sample
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

        latents = (latents / vae.config.scaling_factor).to(weight_dtype)
        images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.float()
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")

    if args.logger == "tensorboard":
        tracker = (
            accelerator.get_tracker("tensorboard", unwrap=True)
            if is_accelerate_version(">=", "0.17.0.dev0")
            else accelerator.get_tracker("tensorboard")
        )
        tracker.add_images("samples", images.transpose(0, 3, 1, 2), global_step)
    elif args.logger == "wandb":
        import wandb

        accelerator.get_tracker("wandb").log(
            {"samples": [wandb.Image(img) for img in images], "epoch": epoch},
            step=global_step,
        )

    if accelerator.is_main_process:
        preview_dir = Path(args.output_dir) / "images"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images[: min(4, len(images))]):
            Image.fromarray(img).save(preview_dir / f"sample_epoch{epoch}_img{idx}.png")


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Install tensorboard if you want to use it for logging during training.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb if you want to use it for logging during training.")
        import wandb  # noqa: F401

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    cond_spec = parse_condition_spec(args.condition_spec)
    condition_encoder = ConditionEncoder(cond_spec)
    cond_dim = condition_encoder.cond_dim

    # Load and freeze VAE (trained from scratch separately).
    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.requires_grad_(False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_channels = args.latent_channels_override or vae.config.latent_channels
    latent_size = args.resolution // vae_scale_factor
    if latent_size * vae_scale_factor != args.resolution:
        logger.warning(
            "Resolution %s is not divisible by VAE scale factor %s; rounding down latent size to %s.",
            args.resolution,
            vae_scale_factor,
            latent_size,
        )

    cross_attention_dim = cond_dim if cond_dim > 0 else 128
    if args.model_config_name_or_path is None:
        model = UNet2DConditionModel(
            sample_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=2,
            block_out_channels=(256, 256, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=cross_attention_dim,
            class_embed_type="projection" if cond_dim > 0 else None,
            projection_class_embeddings_input_dim=cond_dim if cond_dim > 0 else None,
        )
    else:
        config = UNet2DConditionModel.load_config(args.model_config_name_or_path)
        config.update(
            {
                "sample_size": latent_size,
                "in_channels": latent_channels,
                "out_channels": latent_channels,
                "cross_attention_dim": cross_attention_dim,
                "class_embed_type": "projection" if cond_dim > 0 else None,
                "projection_class_embeddings_input_dim": cond_dim if cond_dim > 0 else None,
            }
        )
        model = UNet2DConditionModel.from_config(config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it or disable --enable_xformers_memory_efficient_attention.")

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DConditionModel,
            model_config=model.config,
        )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transform = build_image_transform(
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        preserve_input_precision=args.preserve_input_precision,
    )
    dataset = ManifestDataset(
        manifest_path=Path(args.manifest_path),
        split=args.split,
        transform=transform,
        condition_encoder=condition_encoder,
        limit=args.max_train_samples,
    )
    logger.info("Loaded %s samples from %s split", len(dataset), args.split)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for _ in range(len(models)):
                    unet_ = models.pop()
                    unet_.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                loaded_ema = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNet2DConditionModel,
                )
                ema_model.load_state_dict(loaded_ema.state_dict())
                ema_model.to(accelerator.device)
                del loaded_ema

            for _ in range(len(models)):
                model_ = models.pop()
                loaded_unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model_.register_to_config(**loaded_unet.config)
                model_.load_state_dict(loaded_unet.state_dict())
                del loaded_unet

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running latent training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Effective batch size = {total_batch_size}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    vae.to(accelerator.device, dtype=weight_dtype)
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch:
                if step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                else:
                    args.resume_from_checkpoint = None

            clean_pixels = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            cond_labels = None
            if cond_dim > 0:
                cond_labels = batch["cond"].to(device=accelerator.device, dtype=weight_dtype)
                if args.cond_dropout_prob > 0:
                    drop_mask = torch.rand(cond_labels.shape[0], device=cond_labels.device) < args.cond_dropout_prob
                    cond_labels = cond_labels.clone()
                    cond_labels[drop_mask] = 0

            with torch.no_grad():
                latents = vae.encode(clean_pixels).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.int64,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hs = torch.zeros(
                (bsz, 1, cross_attention_dim),
                device=latents.device,
                dtype=weight_dtype,
            )
            if cond_labels is not None:
                encoder_hs = cond_labels.unsqueeze(1)

            with accelerator.accumulate(model):
                model_output = model(noisy_latents, timesteps, encoder_hidden_states=encoder_hs, class_labels=cond_labels).sample
                loss = F.mse_loss(model_output.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info("Saved state to %s", save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet_for_eval = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet_for_eval.parameters())
                    ema_model.copy_to(unet_for_eval.parameters())

                log_validation_images(
                    accelerator=accelerator,
                    unet=unet_for_eval,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    args=args,
                    weight_dtype=weight_dtype,
                    vae_scale_factor=vae_scale_factor,
                    global_step=global_step,
                    epoch=epoch,
                    cond_dim=cond_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                if args.use_ema:
                    ema_model.restore(unet_for_eval.parameters())

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet_to_save = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet_to_save.parameters())
                    ema_model.copy_to(unet_to_save.parameters())

                unet_to_save.save_pretrained(Path(args.output_dir) / "unet")
                noise_scheduler.save_pretrained(args.output_dir)

                with open(Path(args.output_dir) / "vae_source.json", "w") as f:
                    json.dump(
                        {
                            "vae_path": args.vae_path,
                            "condition_spec": cond_spec,
                        },
                        f,
                        indent=2,
                    )

                if args.use_ema:
                    ema_model.restore(unet_to_save.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
