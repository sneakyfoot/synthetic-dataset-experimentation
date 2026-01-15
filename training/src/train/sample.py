import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image

from train.data import parse_condition_spec


def _load_cond_values(cond_json: Optional[str]) -> Dict:
    if cond_json is None:
        return {}
    with open(cond_json, "r") as f:
        return json.load(f)


def _build_cond_tensor(cond_spec: List, cond_values: Dict, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if not cond_spec:
        return None

    values = []
    for name, dim in cond_spec:
        raw = cond_values.get(name)
        if name == "wind_mag" and raw is None:
            wind = cond_values.get("wind")
            if isinstance(wind, (list, tuple)):
                raw = float(sum(v * v for v in wind) ** 0.5)

        if raw is None:
            values.extend([0.0] * dim)
            continue

        if isinstance(raw, (list, tuple)):
            padded = list(raw)[:dim]
            padded.extend([0.0] * (dim - len(padded)))
            values.extend([float(v) for v in padded])
        else:
            try:
                scalar = float(raw)
            except (TypeError, ValueError):
                scalar = 0.0
            values.append(scalar)
            if dim > 1:
                values.extend([0.0] * (dim - 1))

    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from a trained latent DDPM.")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to UNet weights (folder containing config.json).")
    parser.add_argument("--vae_path", type=str, required=False, help="Path to VAE weights. If omitted, tries <unet_path>/../vae_source.json.")
    parser.add_argument("--scheduler_path", type=str, default=None, help="Path to scheduler config (defaults to parent of UNet).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save samples.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--condition_spec", type=str, default=None, help='Comma list "name:dim" (default loads from vae_source.json if present). Use "none" for unconditional.')
    parser.add_argument("--cond_json", type=str, default=None, help="JSON file with condition values {name: value or list}.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Compute dtype.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    scheduler_path = args.scheduler_path or str(Path(args.unet_path).parent)

    # Resolve VAE path and default condition_spec from vae_source.json if present.
    vae_path = args.vae_path
    vae_source = Path(args.unet_path).parent / "vae_source.json"
    if vae_path is None and vae_source.exists():
        with open(vae_source, "r") as f:
            meta = json.load(f)
            vae_path = meta.get("vae_path", vae_path)
            if args.condition_spec is None:
                args.condition_spec = meta.get("condition_spec")
    if vae_path is None:
        raise ValueError("VAE path not provided and vae_source.json not found.")

    cond_spec = parse_condition_spec(args.condition_spec)
    cond_values = _load_cond_values(args.cond_json)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16 if args.dtype == "bf16" else torch.float32

    vae = AutoencoderKL.from_pretrained(vae_path).to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=dtype).to(device)
    scheduler = DDPMScheduler.from_pretrained(scheduler_path)

    scheduler.set_timesteps(args.num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    latent_size = unet.config.sample_size
    latent_shape = (args.num_samples, unet.config.in_channels, latent_size, latent_size)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    class_labels = _build_cond_tensor(cond_spec, cond_values, device=device, dtype=dtype)
    encoder_hs = torch.zeros(
        (args.num_samples, 1, getattr(unet.config, "cross_attention_dim", 128)),
        device=device,
        dtype=dtype,
    )

    for t in scheduler.timesteps:
        noise_pred = unet(latents, t, encoder_hidden_states=encoder_hs, class_labels=class_labels).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1).float()

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255).round().astype("uint8")
        Image.fromarray(img).save(Path(args.output_dir) / f"sample_{i}.png")


if __name__ == "__main__":
    main()
