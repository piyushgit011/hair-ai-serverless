#!/bin/bash


# Install PyTorch
git clone https://github.com/IDEA-Research/GroundingDINO
cd GroundingDINO
pip install -e .
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..

git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install timm torch flask  
mkdir pretrained_checkpoint
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth -O pretrained_checkpoint/sam_hq_vit_l.pth
export PYTHONPATH=$(pwd)
cd ..
