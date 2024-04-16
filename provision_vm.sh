#!/bin/bash
apt-get install python3.9-venv
update-alternatives --install /usr/bin/python3 python /usr/bin/python3.9 1
python3 -m venv venv
source venv/bin/activate

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
apt install python3.9-dev build-essential python3-wheel -y
apt-get install portaudio19-dev python-all-dev
apt-get install python3-pyaudio

pip3 install wheel
pip3 install boto3 awscli numpy
pip3 install opencv-python

pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

git clone git@github.com:tom-bbc/video-testing-automation.git
cd video-testing-automation
git submodule update --init --recursive
pip install -r requirements.txt
pip install -r ExplainableVQA/requirements.txt
sed -i '92s/return x\[0\]/return x/' ExplainableVQA/open_clip/src/open_clip/modified_resnet.py
pip install -e ExplainableVQA/open_clip
pip install -e ExplainableVQA/DOVER
pip install wandb
mkdir ExplainableVQA/DOVER/pretrained_weights
wget https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth -P ExplainableVQA/DOVER/pretrained_weights/
