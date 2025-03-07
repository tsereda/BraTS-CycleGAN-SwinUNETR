# Start with the base PRP image
FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp

# Set the working directory
WORKDIR /opt

# Install git
USER root
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone --depth=1 https://gitlab.nrp-nautilus.io/timothy.sereda/BraTS-CycleGAN-SwinUNETR /opt/BraTS-CycleGAN-SwinUNETR

# Switch back to the jovyan user
USER jovyan

# Create and activate conda environment from the repo's environment file
# Creating it at the system level means we don't need a persistent volume for conda environments
RUN conda env create -f /opt/BraTS-CycleGAN-SwinUNETR/environment.yml -n BraTS && \
    conda clean -afy

# Add the conda initialization to .bashrc
RUN echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc && \
    echo 'conda activate BraTS' >> ~/.bashrc

# Fix the split-folders package issue
RUN conda activate BraTS && pip install split-folders

# Create data directory for the BraTS dataset
RUN mkdir -p /data

# Copy your dataset in - comment this out if you plan to mount the dataset instead
# COPY ./brats_dataset/ /data/

# Set the entrypoint to use the conda environment
ENTRYPOINT ["/bin/bash", "-c", "eval \"$(conda shell.bash hook)\" && conda activate BraTS && exec \"$@\"", "--"]

# Default command (can be overridden)
CMD ["python", "/opt/BraTS-CycleGAN-SwinUNETR/data_preprocessing.py"]