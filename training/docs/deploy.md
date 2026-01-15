# Deployment (Kubernetes)

This repo now uses Job-based distributed runs (no StatefulSet loop). Two manifests:

- `deploy/train_job.yaml`: latent DDPM training (stage 2).
- `deploy/train_job_vae.yaml`: VAE training (stage 1).

Both:
- Use headless Service `torchtrain` for rendezvous DNS.
- `parallelism = completions = 6`, `completionMode: Indexed`; `JOB_COMPLETION_INDEX` becomes the machine rank.
- Rendezvous host names match the Job names: `torchtrain-latent-0.torchtrain.ml.svc.cluster.local` and `torchtrain-vae-0.torchtrain.ml.svc.cluster.local`.
- VAE job defaults aimed at higher fidelity: `block_out_channels=128,256,512,512`, `latent_channels=4`, `kl_weight=1e-6`, batch size 6, LPIPS enabled with `--lpips_cache /mnt/RAID/torch_cache` (preload weights there to avoid downloads).
- Latent job expects the matching latent channels via `--latent_channels_override 8`.
- Host networking; NCCL interface is set to `eth` by default—override via env if needed.
- Mount `/mnt/RAID` and `/dev/shm`.

Latent job (`train_job.yaml`):
- Runs `accelerate launch --module train.latent_diffusion` with your manifest + VAE paths.
- Default flags: resolution 256, condition spec `temperature:1,wind:3,wind_mag:1`, batch sizes 4, GAcc 2, EMA on, bf16.

VAE job (`train_job_vae.yaml`):
- Runs `accelerate launch --module train.vae`.
- Default flags: resolution 256, batch 8, bf16, save images every epoch, checkpoints every 1000 steps.

Adjust `parallelism/completions`, batch sizes, and paths as needed before applying. Apply with:
```bash
kubectl apply -f deploy/train_job_vae.yaml
kubectl apply -f deploy/train_job.yaml
```
