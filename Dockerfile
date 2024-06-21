# Use the official PyTorch image with CUDA 11.8
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV CUDA_HOME /usr/local/cuda
# Set the working directory in the container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools
RUN apt-get update && \
    apt-get install -y git wget build-essential cmake && \
    pip install --no-cache-dir timm torch flask torchvision

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install PyTorch
RUN pip install Cython numpy==1.21.2

# Clone and install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO && \
    cd GroundingDINO && \
    pip install -e . && \
    mkdir weights && \
    cd weights && \
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
    cd ../..

# Clone and install SAM-HQ
RUN git clone https://github.com/SysCV/sam-hq.git && \
    cd sam-hq && \
    pip install timm torch flask && \
    mkdir pretrained_checkpoint && \
    wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth -O pretrained_checkpoint/sam_hq_vit_l.pth && \
    export PYTHONPATH=$(pwd) && \
    cd ..

ENV PYTHONPATH="/app/sam-hq:$PYTHONPATH"

# Copy the service account key file to the container

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/Hair AI Firebase Admin.json"

# Copy all files from the current directory to /app in the container
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

RUN pip install numpy==1.23.0 

RUN pip install opencv-python==4.8.0.74

# Command to run your application
CMD ["python3", "-u", "rp_handler.py"]
