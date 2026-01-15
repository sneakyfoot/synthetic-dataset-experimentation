# VAE Training (from scratch)

Command (example):
```bash
python -m train.vae \
  --manifest_path /mnt/RAID/Assets/Torch_Dataset_001/beauty_all-TORCH-001-ML-15.json \
  --split train \
  --output_dir /mnt/RAID/Assets/Torch_Dataset_001/vae \
  --resolution 256 \
  --train_batch_size 8 \
  --num_epochs 50 \
  --mixed_precision fp16 \
  --save_images_epochs 5
```

Model:
- `AutoencoderKL` with configurable `block_out_channels` (default `128,256,256,256`), `layers_per_block = 2`.
- Latent channels: configurable (default 4). Downsample factor: 4 levels → ×8 (256 → 32×32); 3 levels → ×4 (256 → 64×64).
- Scaling factor defaults to 1.0 in this from-scratch setup.

Loss:
- Reconstruction = MSE + L1 (pixel space).
- KL term with weight `--kl_weight` (default 1e-4).

Logging & previews:
- Recon pairs (GT/recon) saved to `output_dir/images/` every `save_images_epochs` and on final epoch.
- Also logged to TensorBoard/W&B if enabled.

Outputs:
- Diffusers-style VAE in `output_dir/`.
- Images for visual monitoring in `output_dir/images/`.

Precision:
- `--mixed_precision fp16` or `bf16` (recommended if GPUs support bf16). We cast the model to the chosen dtype post-accelerate to avoid bias/input mismatches.

Checkpointing/resume:
- Use `--checkpointing_steps N` to save training state (model + optimizer + scheduler) every N steps, with `--checkpoints_total_limit` to cap how many to keep.
- Resume with `--resume_from_checkpoint <path>` or `--resume_from_checkpoint latest` to auto-pick the newest `checkpoint-*` under `output_dir`.

VRAM guidance (~20 GB target):
- Start with `train_batch_size=8` at 256². If OOM, lower batch size first, then consider fewer channels (e.g., `block_out_channels=128,192,256`) or reduce `resolution`. If you have headroom and want sharper reconstructions, try `block_out_channels=128,256,256` (×4) and `--latent_channels 8` for higher fidelity.
