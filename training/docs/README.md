# Torch Training Docs

This repo trains a diffusion model **from scratch** on a single Houdini sim dataset described by the manifest file. The pipeline has two stages:

1. Train a VAE (`train.vae`) on the manifest images (unconditional).
2. Freeze the VAE and train a latent DDPM (`train.latent_diffusion`), optionally conditioning on numeric vectors (temperature, wind, etc.).

Quick links:
- [Data & manifest](data.md)
- [VAE training](vae.md)
- [Latent DDPM training](latent.md)
- [Inference & sampling](inference.md)
- [Troubleshooting & VRAM tips](troubleshooting.md)
