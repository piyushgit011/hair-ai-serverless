# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0
# Install PyTorch
RUN pip install torch torchvision torchaudio

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

# Copy all files from the current directory to /app in the container
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Command to run your application
CMD ["python3", "-u", "rp_handler.py"]
