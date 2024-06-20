tmux new
---------------------------------
cd workspace
git clone https://github.com/piyushgit011/groundingdino_sam_api.git
cd groundingdino_sam_api
rm -rf GroundingDINO/ MobileSAM/
git clone https://github.com/piyushgit011/MobileSAM.git
git clone https://github.com/IDEA-Research/GroundingDINO
cd GroundingDINO
pip3 install torch torch vision torch audio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
cd ../MobileSAM
pip install -e .
cd ..
pip install -r requirements.txt
pip install boto3
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
./download_weights.sh
cd ..
python3 main.py
on keyboard select: ctrl + b d

git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install timm torch flask  
mkdir pretrained_checkpoint
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth -O pretrained_checkpoint/sam_hq_vit_l.pth
export PYTHONPATH=$(pwd)
cd ..
