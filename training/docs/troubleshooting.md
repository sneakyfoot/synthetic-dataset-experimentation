# Troubleshooting & VRAM Tips

Common issues:
- **Pickling error on DataLoader workers**: already addressed by replacing lambda transforms with picklable no-ops. If it reappears, set `--dataloader_num_workers 0`.
- **Dtype mismatch (conv bias vs fp16 inputs)**: ensure you’re on the latest code; the VAE is cast to the mixed-precision dtype after accelerator setup.
- **Missing image warnings**: the manifest points to a file that is not present. Either restore it or let the loader skip; dataset size will shrink accordingly.

VRAM management (~20 GB goal):
- Lower `train_batch_size` first.
- If still high:
  - VAE stage: consider narrowing `block_out_channels` (code change) or lowering `resolution`.
  - Latent stage: reduce UNet width (e.g., `(192,192,256)`) or lower `resolution`.
- Keep `--enable_xformers_memory_efficient_attention` on (latent stage) if installed.
- Use `--mixed_precision bf16` if your GPUs handle bf16 efficiently.

Monitoring:
- VAE: recon pairs saved to `output_dir/images/`; adjust `--save_images_epochs` for cadence.
- Latent: decoded samples saved to `output_dir/images/`; same cadence flag.
- TensorBoard/W&B logging is enabled when `--logger tensorboard|wandb` is set.
