# synthetic-dataset-experimentation
Experiments generating synthetic data from houdini and using it to train diffusion models. 

From Houdini, I generated a dataset of about 10,000 sample images, each with a unique source /force seeds but otherwise with identical simulation settings. This took about 3 days to generate on my cluster. 

I was curious if this amount of data would be enough to train diffusion models, and how closeley the images generated at inference time would match my dataset.

### Example sample from the dataset
<img width="256" height="256" alt="dataset" src="https://github.com/user-attachments/assets/a4eef1c0-50ef-4289-9759-8d940c314e7c" />

I first tried using the huggingface train_unconditional_diffusion.py example file as a basic proof of concept.

## Unconditional pixel space diffusion
### Example output from the unconditional diffusion model
<img width="256" height="256" alt="sample_image_8" src="https://github.com/user-attachments/assets/9b178f01-aee5-4023-8784-befbc1633ba1" />
<img width="256" height="256" alt="sample_image_7" src="https://github.com/user-attachments/assets/47735b60-7eb3-4dfd-8948-d0f31dc1f6d4" />

Pretty neat! The results look very close to my dataset, although inference takes about 10x as long as the renders out of Houdini. 

## Latent space diffusion with VAE
In order to add conditional vectors, ie. wind, temperature etc. at inference time, I moved to a two stage latent diffusion workflow. Before training a new dataset with varations across different parameters, I tried training the VAE and diffusion model with the same dataset, to see if I could even get a good reconstruction from the encoder/decoder.

### VAE
- Reconstruction going from pixel space to latent space and back
<img width="256" height="256" alt="epoch80_step28674_img6" src="https://github.com/user-attachments/assets/4c6e228d-76fd-4b23-b1d1-6e17dd86dfcb" />
<img width="256" height="256" alt="epoch80_step28674_img7" src="https://github.com/user-attachments/assets/90ffdc03-3e57-40fd-a544-6f4a1274a168" />

Not too bad, after adjusting the the loss metric to use LPIP, I was able to get reasonable reconstruction, although there is an obvious loss in fedelity. 

### Generations
- Images generated after training the latent space diffusion model
<img width="256" height="256" alt="sample_epoch49_img1" src="https://github.com/user-attachments/assets/722956b4-b65e-4bcc-9b9a-dd4b1d32f248" />
<img width="256" height="256" alt="sample_epoch49_img3" src="https://github.com/user-attachments/assets/2dd36afc-c587-4d58-bfa8-d7ef32e522bc" />

We see the same fedelity loss compared to the origional dataset, but with latent diffusion, inference is much faster, and the blacks from the background are preserved much better, and overall a much less grainy output.
