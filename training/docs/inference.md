# Inference & Sampling

## CLI sampler
Use the bundled script to sample and save PNGs:
```bash
sample-latent \
  --unet_path /mnt/RAID/Assets/Torch_Dataset_001/latent-ddpm/unet \
  --vae_path /mnt/RAID/Assets/Torch_Dataset_001/vae \
  --output_dir /mnt/RAID/Assets/Torch_Dataset_001/samples \
  --num_samples 4 \
  --num_inference_steps 50 \
  --dtype fp16 \
  --condition_spec "temperature:1,wind:3,wind_mag:1" \
  --cond_json /path/to/conds.json
```
- If `--vae_path` is omitted, the script tries `<unet_parent>/vae_source.json` to resolve it (and default `condition_spec`).
- `--cond_json` should contain a dict of values, e.g. `{"temperature":0.2,"wind":[0.1,0.0,0.0]}`; `wind_mag` auto-computes if omitted. Use `--condition_spec none` for unconditional (or omit `cond_json`).
- Dtypes: `fp16`, `bf16`, `fp32`. Device defaults to `cuda`.

## Manual snippet (unconditional)
```python
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

device = "cuda"
vae = AutoencoderKL.from_pretrained("/mnt/RAID/Assets/Torch_Dataset_001/vae").to(device, torch.float16)
unet = UNet2DConditionModel.from_pretrained("/mnt/RAID/Assets/Torch_Dataset_001/latent-ddpm/unet", torch_dtype=torch.float16).to(device)
scheduler = DDPMScheduler.from_pretrained("/mnt/RAID/Assets/Torch_Dataset_001/latent-ddpm")

cond_dim = getattr(unet.config, "projection_class_embeddings_input_dim", 0) or 0
cond = torch.zeros((1, cond_dim), device=device, dtype=torch.float16) if cond_dim > 0 else None
encoder_hs = torch.zeros((1, 1, getattr(unet.config, "cross_attention_dim", cond_dim or 128)), device=device, dtype=torch.float16)

scheduler.set_timesteps(50)
latent_size = unet.config.sample_size
latents = torch.randn((1, unet.config.in_channels, latent_size, latent_size), device=device, dtype=torch.float16)
for t in scheduler.timesteps:
    noise_pred = unet(latents, t, encoder_hidden_states=encoder_hs, class_labels=cond).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = latents / vae.config.scaling_factor
images = vae.decode(latents).sample
images = (images / 2 + 0.5).clamp(0, 1)
image = images[0].permute(1, 2, 0).cpu().numpy()
```

## Conditioning tips
- Build the vector in the order specified by `condition_spec`. For the default spec: `[temperature, wind_x, wind_y, wind_z, wind_mag]`.
- Missing values are zero-filled; `wind_mag` auto-computes from `wind` if absent.
- During sampling, `cond_dropout_prob` is not applied—pass full vectors or zeros for unconditional output.
- The condition vector is passed as a single token via `encoder_hidden_states`; keep the dimension equal to the sum of your `condition_spec` dims.
