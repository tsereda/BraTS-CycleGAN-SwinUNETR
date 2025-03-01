First, log into HPC

```bash
# Start a session on a high memory node (only for env create and download_data)
srun --pty -p himem bash

# Clone this repo
git clone https://github.com/tsereda/BraTS-CycleGAN-SwinUNETR.git
cd BraTS-CycleGAN+SwinUNETR

# Create conda environment from the provided environment.yml
conda env create -f environment.yml

# Make directory
mkdir slurm_logs

# Download data to shared/kaggle_cache and copy training data
python download_data.py

# Submit data preprocessing job
sbatch data_preprocessing.sh

# Train
sbatch train_swinunetr.sh
```


bonus commands

conda env remove -n BraTS