# Latent DDPM Training (from scratch, numeric conditioning-ready)

Command (example):
```bash
python -m train.latent_diffusion \
  --manifest_path /mnt/RAID/Assets/Torch_Dataset_001/beauty_all-TORCH-001-ML-15.json \
  --vae_path /mnt/RAID/Assets/Torch_Dataset_001/vae \
  --output_dir /mnt/RAID/Assets/Torch_Dataset_001/latent-ddpm \
  --resolution 256 \
  --condition_spec "temperature:1,wind:3,wind_mag:1" \
  --cond_dropout_prob 0.1 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --num_epochs 50 \
  --use_ema \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention
```

Model:
- Latent UNet2DConditionModel initialized from scratch. Default `block_out_channels = (256, 256, 256)`, `layers_per_block = 2`.
- Latent shape depends on VAE: 4 levels → ×8 (e.g., `[B, latent_ch, 32, 32]` at 256²); 3 levels → ×4 (e.g., `[B, latent_ch, 64, 64]`).
- Conditioning: numeric vector passed two ways: (a) `class_labels` via projection head (`projection_class_embeddings_input_dim=cond_dim`), and (b) a single-token `encoder_hidden_states` (zero or cond token) for cross-attention. `cond_dropout_prob` zeroes `class_labels` for classifier-free style.

Scheduler & loss:
- DDPM scheduler (linear betas), `ddpm_num_steps=1000`. Training prediction target: noise (MSE).
- EMA optional via `--use_ema`.

Logging & previews:
- Decoded samples saved to `output_dir/images/` every `save_images_epochs` and on final epoch; logged to TensorBoard/W&B if enabled.

Outputs:
- `output_dir/unet/` (UNet weights), scheduler config in `output_dir/`, and `vae_source.json` noting the VAE path and condition spec.

VRAM guidance (~20 GB target):
- Reduce `train_batch_size` first if OOM.
- If still high, simplify UNet width, e.g., `block_out_channels=(192,192,256)` (requires code edit) or lower resolution.
- `--mixed_precision bf16` is supported if your GPUs handle it well; keep `--enable_xformers_memory_efficient_attention` on for memory savings when available.
