Log into HPC

```bash
# Start a session on a non-login node (faster conda env create and download)
srun --pty -p himem bash

# Clone this repo
git clone htps://github.com/this
cd BraTS-CycleGAN+SwinUNETR

# Create conda environment from the provided environment.yml
conda env create -f environment.yml

# Make directory
mkdir slurm_logs

# Download data to shared/kaggle_cache and copy training data
python download_data.py

# Submit data preprocessing job
sbatch data_preprocessing.sh


```


bonus commands
conda env remove -n BraTS

