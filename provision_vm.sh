#!/bin/bash
apt install python3.9
update-alternatives --install /usr/bin/python3 python /usr/bin/python3.9 1
apt-get install python3.9-venv
python3 -m venv venv
source venv/bin/activate

pip3 install boto3
pip3 install awscli
pip3 install numpy

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
pip3 install opencv-python
apt install python3.9-dev build-essential python3.9-wheel -y
pip3 install wheel
python3 -m  pip install PyAudio

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall

git clone git@github.com:tom-bbc/video-testing-automation.git
