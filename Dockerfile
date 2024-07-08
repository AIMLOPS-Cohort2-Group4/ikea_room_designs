# Base image with CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install the NVIDIA Container Toolkit
RUN apt-get update && \
    apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Set up the Python environment
RUN pip3 install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY . /app
WORKDIR /app

# Expose the port the app runs on
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]