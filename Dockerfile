# Use the specified base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Install the necessary packages
RUN conda install -qy pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg && \
    conda clean -afy

# Copy the current directory contents into the container
COPY . /tmp
RUN pip install -e git+https://github.com/Open-Catalyst-Project/ocp.git@main#egg=ocp-models
RUN pip install /tmp[dev] && \
    rm -rf /root/.cache/pip/*
