
Log into HPC



```bash
# Start a session on a non-login node
srun --pty bash

# Clone this repo
git clone htps://github.com/this
cd BraTS-CycleGAN+SwinUNETR

# Create conda environment from the provided environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate pytorch-BraTS2020-unet-segmentation

# Verify GPU support
python gpucheck.py
```
