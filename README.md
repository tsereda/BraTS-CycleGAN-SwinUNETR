
Log into HPC



```bash
# Start a session on a non-login node (faster conda env create and download)
srun --pty bash
#srun --pty -p nodes bash?

# Clone this repo
git clone htps://github.com/this
cd BraTS-CycleGAN+SwinUNETR

# Create conda environment from the provided environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate BraTS


```


bonus commands
conda env remove -n BraTS