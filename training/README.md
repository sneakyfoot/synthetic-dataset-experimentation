See `docs/` for detailed guidance. Legacy DDPM (pixel-space) training is preserved under `src/legacy/ddpm.py` with a console script `train-legacy-ddpm`.

## From-scratch pipeline (manifest → VAE → latent DDPM)

### 1) Train a VAE from scratch on the manifest
```bash
python -m train.vae \
  --manifest_path /mnt/RAID/Assets/Torch_Dataset_001/beauty_all-TORCH-001-ML-15.json \
  --split train \
  --output_dir /mnt/RAID/Assets/dataset_test/vae \
  --resolution 256 \
  --train_batch_size 8 \
  --num_epochs 50 \
  --mixed_precision fp16
```
- Uses `AutoencoderKL` with latent_channels=4, downsample ×8 (256 → 32 latents).  
- Loss: MSE + L1 + KL (weight `--kl_weight`, default 1e-4).  
- Recon samples saved to `output_dir/images/`; weights saved in diffusers format at `output_dir`.
- Preview cadence: `--save_images_epochs` (default 5). Set to 1 for per-epoch snapshots. Logged to TensorBoard/W&B when enabled.
- Dtype: `--mixed_precision bf16` is supported if your GPUs prefer it. Model is cast to the chosen dtype post-accelerate setup to avoid bias/input mismatches.

### 2) Train a latent DDPM from scratch (optional conditioning ready)
```bash
python -m train.latent_diffusion \
  --manifest_path /mnt/RAID/Assets/Torch_Dataset_001/beauty_all-TORCH-001-ML-15.json \
  --vae_path /mnt/RAID/Assets/dataset_test/vae \
  --output_dir /mnt/RAID/Assets/dataset_test/latent-ddpm \
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
- `condition_spec` is a comma list of `name:dim`; use `none` to run fully unconditional. Missing metadata values are zero-filled so you can start training before conditions land in the manifest.  
- Conditions are fed via a projection head into the UNet’s class embedding; dropout (`--cond_dropout_prob`) provides classifier-free compatibility.  
- Saved artifacts: `output_dir/unet/`, scheduler config in `output_dir/`, `vae_source.json` noting the VAE and condition spec, sample previews in `output_dir/images/`.
- Typical latent shapes with resolution 256 and VAE ×8 downsample: latents are `[B, 4, 32, 32]`. Default UNet blocks `(256,256,256)` keep VRAM moderate; adjust `block_out_channels` and batch size to hit your ~20 GB target.

### 3) Sampling (unconditional or zeros for conditions)
```python
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

device = "cuda"
vae = AutoencoderKL.from_pretrained("/mnt/RAID/Assets/dataset_test/vae").to(device, torch.float16)
unet = UNet2DModel.from_pretrained("/mnt/RAID/Assets/dataset_test/latent-ddpm/unet", torch_dtype=torch.float16).to(device)
scheduler = DDPMScheduler.from_pretrained("/mnt/RAID/Assets/dataset_test/latent-ddpm")

cond_dim = unet.config.projection_class_embeddings_input_dim or 0
cond = torch.zeros((1, cond_dim), device=device, dtype=torch.float16) if cond_dim > 0 else None

scheduler.set_timesteps(50)
latent_size = unet.config.sample_size
latents = torch.randn((1, unet.config.in_channels, latent_size, latent_size), device=device, dtype=torch.float16)
for t in scheduler.timesteps:
    noise_pred = unet(latents, t, class_labels=cond).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = latents / vae.config.scaling_factor
images = vae.decode(latents).sample
images = (images / 2 + 0.5).clamp(0, 1)
images[0].permute(1, 2, 0).cpu().numpy()
```

### Notes on the manifest loader and conditions
- Loader reads `output_rel` paths and filters by `metadata.split`. Missing files are skipped with a warning.
- Default condition spec: `temperature:1,wind:3,wind_mag:1`. `wind_mag` auto-computes from `wind` when absent; any missing value is zero-filled. Custom specs via `--condition_spec name:dim,...`; use `none` for unconditional.
- Transforms preserve 16/32-bit precision when `--preserve_input_precision` is set; otherwise standard `ToTensor` + normalize to [-1,1].
