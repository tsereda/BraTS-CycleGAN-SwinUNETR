# Start with a minimal Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /opt/app

# Install system dependencies (minimized)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda instead of full Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Copy local code to the container
COPY . /opt/app/

# Create conda environment from the environment file
RUN conda env create -f environment.yml

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "BraTS", "/bin/bash", "-c"]

# Install additional Python dependencies
RUN pip install split-folders

# Clean conda to reduce image size
RUN conda clean -afy && \
    find /opt/conda/ -type f -name '*.a' -delete && \
    find /opt/conda/ -type f -name '*.js.map' -delete && \
    find /opt/conda/ -type f -name '*.pyc' -delete && \
    find /opt/conda/ -type f -name '*.c' -delete

# Create directories for data and output
RUN mkdir -p /opt/app/brats20-dataset /opt/app/processed_data /opt/app/logs /opt/app/output

# Set default command
CMD ["conda", "run", "--no-capture-output", "-n", "BraTS", "python", "segmentation/train.py"]