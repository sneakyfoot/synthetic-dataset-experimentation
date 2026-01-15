import argparse
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
try:
    import lpips  # type: ignore
except Exception:
    lpips = None
import shutil
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from train.data import ManifestDataset, build_image_transform
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    is_tensorboard_available,
    is_wandb_available,
)
from PIL import Image
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an AutoencoderKL (VAE) from scratch on the manifest dataset.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the dataset manifest JSONL file.",
    )
    parser.add_argument("--split", type=str, default="train", help="Manifest split to train on.")
    parser.add_argument("--output_dir", type=str, default="vae-model", help="Where to write checkpoints and samples.")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--preserve_input_precision", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_images_epochs", type=int, default=5)
    parser.add_argument("--save_model_epochs", type=int, default=5)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="If >0, save training state every N optimizer steps (for resuming mid-epoch).",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to keep when using checkpointing_steps.",
    )
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
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="KL term weight.")
    parser.add_argument("--latent_channels", type=int, default=4, help="Latent channels for the VAE.")
    parser.add_argument(
        "--block_out_channels",
        type=str,
        default="128,256,256,256",
        help="Comma list of channel widths per encoder/decoder level.",
    )
    parser.add_argument("--l1_weight", type=float, default=1.0, help="Weight for L1 reconstruction loss.")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Weight for MSE reconstruction loss.")
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=0.0,
        help="Weight for LPIPS perceptual loss (set >0 to enable). Requires lpips package.",
    )
    parser.add_argument(
        "--lpips_cache",
        type=str,
        default=None,
        help="Optional path to use as TORCH_HOME for LPIPS weights (pre-download weights here to avoid network).",
    )
    parser.add_argument(
        "--vae_scaling_factor",
        type=float,
        default=1.0,
        help="Scaling factor applied to latents (keep 1.0 for simplicity in from-scratch training).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Path to a checkpoint to resume from, or "latest" to auto-pick the newest under output_dir.',
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def log_reconstructions(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    batch,
    epoch: int,
    global_step: int,
    weight_dtype: torch.dtype,
    output_dir: Path,
):
    vae.eval()
    vae_for_fwd = vae.module if hasattr(vae, "module") else vae
    with torch.no_grad():
        images = batch[: min(4, batch.shape[0])].to(device=accelerator.device, dtype=weight_dtype)
        latents = vae_for_fwd.encode(images).latent_dist.sample()
        latents = latents * vae_for_fwd.config.scaling_factor
        recon = vae_for_fwd.decode(latents / vae_for_fwd.config.scaling_factor).sample
        recon = (recon / 2 + 0.5).clamp(0, 1)
        images = (images / 2 + 0.5).clamp(0, 1)
        # Ensure numpy conversion works even under bf16.
        recon = recon.float()
        images = images.float()

    recon_cpu = recon.detach().cpu().permute(0, 2, 3, 1).numpy()
    images_cpu = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    stacked = []
    for gt, rc in zip(images_cpu, recon_cpu):
        stacked.append((gt * 255).round().astype("uint8"))
        stacked.append((rc * 255).round().astype("uint8"))

    if accelerator.is_main_process:
        preview_dir = output_dir / "images"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(stacked):
            Image.fromarray(img).save(preview_dir / f"epoch{epoch}_step{global_step}_img{idx}.png")

    if accelerator.log_with == "tensorboard":
        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        tracker.add_images(
            "reconstructions",
            torch.tensor(stacked).permute(0, 3, 1, 2),
            global_step,
        )
    elif accelerator.log_with == "wandb":
        import wandb

        accelerator.get_tracker("wandb").log(
            {"reconstructions": [wandb.Image(img) for img in stacked], "epoch": epoch},
            step=global_step,
        )

    vae.train()


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
            raise ImportError("Install tensorboard for logging.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb for logging.")
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

    block_out_channels = (128, 256, 256, 256)
    if args.block_out_channels:
        block_out_channels = tuple(int(x) for x in args.block_out_channels.split(",") if x.strip())
        if len(block_out_channels) < 2:
            raise ValueError("block_out_channels must have at least 2 entries.")
    down_block_types = ("DownEncoderBlock2D",) * len(block_out_channels)
    up_block_types = ("UpDecoderBlock2D",) * len(block_out_channels)

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=2,
        latent_channels=args.latent_channels,
        sample_size=args.resolution,
        scaling_factor=args.vae_scaling_factor,
        norm_num_groups=32,
    )

    # Decide dtype before optimizer/prepare so parameters and buffers match inputs.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=weight_dtype)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
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
        condition_encoder=None,
        limit=args.max_train_samples,
    )
    logger.info("Loaded %s samples for VAE training", len(dataset))

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

    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    # After prepare, trust the dtype choice above for casting parameters.
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    perceptual_loss = None
    if args.lpips_weight > 0:
        if lpips is None:
            raise ImportError("lpips is not installed. Install it or set --lpips_weight 0.")
        if args.lpips_cache:
            os.environ["TORCH_HOME"] = args.lpips_cache
        # Ensure certs are discoverable for weight download.
        if "SSL_CERT_FILE" not in os.environ and "REQUESTS_CA_BUNDLE" not in os.environ:
            for candidate in ("/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/certs/ca-bundle.crt"):
                if os.path.exists(candidate):
                    os.environ.setdefault("SSL_CERT_FILE", candidate)
                    os.environ.setdefault("REQUESTS_CA_BUNDLE", candidate)
                    break
        try:
            perceptual_loss = lpips.LPIPS(net="vgg").to(accelerator.device)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LPIPS requested but initialization failed (%s). Disabling LPIPS for this run. "
                "Provide predownloaded weights or set --lpips_weight 0 to silence.",
                e,
            )
            args.lpips_weight = 0.0
            perceptual_loss = None

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running VAE training *****")
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

    for epoch in range(args.num_epochs):
        vae.train()
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

            pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

            vae_for_fwd = vae.module if hasattr(vae, "module") else vae

            with accelerator.accumulate(vae):
                posterior = vae_for_fwd.encode(pixel_values).latent_dist
                latents = posterior.sample()
                latents = latents * vae_for_fwd.config.scaling_factor
                recon = vae_for_fwd.decode(latents / vae_for_fwd.config.scaling_factor).sample

                recon_l1 = F.l1_loss(recon.float(), pixel_values.float())
                recon_mse = F.mse_loss(recon.float(), pixel_values.float())
                recon_loss = args.l1_weight * recon_l1 + args.mse_weight * recon_mse
                if perceptual_loss is not None and args.lpips_weight > 0:
                    lp = perceptual_loss(recon, pixel_values).mean()
                    recon_loss = recon_loss + args.lpips_weight * lp
                kl_loss = posterior.kl().mean()
                loss = recon_loss + args.kl_weight * kl_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "recon_loss": recon_loss.detach().item(),
                "recon_l1": recon_l1.detach().item(),
                "recon_mse": recon_mse.detach().item(),
                "kl_loss": kl_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if perceptual_loss is not None and args.lpips_weight > 0:
                logs["lpips"] = lp.detach().item()  # type: ignore[arg-type]
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                log_reconstructions(
                    accelerator=accelerator,
                    vae=vae,
                    batch=pixel_values,
                    epoch=epoch,
                    global_step=global_step,
                    weight_dtype=weight_dtype,
                    output_dir=Path(args.output_dir),
                )

            if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(checkpoint_path)
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) > args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                        for rem in checkpoints[:num_to_remove]:
                            shutil.rmtree(os.path.join(args.output_dir, rem))
                accelerator.print(f"Saved checkpoint to {checkpoint_path}")

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                vae_to_save = vae.module if hasattr(vae, "module") else vae
                vae_to_save.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
