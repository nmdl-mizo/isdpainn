# Use the specified base image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git g++ wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container
COPY . /tmp

# Install the necessary packages
RUN wget https://raw.githubusercontent.com/FAIR-Chem/fairchem/main/packages/env.gpu.yml -P /tmp
RUN conda env update -n base -f /tmp/env.gpu.yml && \
    conda clean -afy
RUN pip install fairchem
RUN pip install /tmp[dev] && \
    rm -rf /root/.cache/pip/*
