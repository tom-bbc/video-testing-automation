#!/bin/bash

# Git submodules
git submodule update --init --recursive

# Python environment requirements
brew install pyenv
pyenv install -v 3.10
pyenv global 3.10
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# Camera capture library
wget https://github.com/matthutchinson/videosnap/releases/download/v0.0.9/videosnap-0.0.9.pkg
sudo installer -pkg videosnap-0.0.9.pkg -target /

# Stutter detection requirements
pip install -r stutter_detection/ExplainableVQA/requirements.txt
sed -i "" "92s/return x\[0\]/return x/" stutter_detection/ExplainableVQA/open_clip/src/open_clip/modified_resnet.py
pip install -e stutter_detection/ExplainableVQA/open_clip
sed -i "" "4s/decord/eva-decord/" stutter_detection/ExplainableVQA/DOVER/requirements.txt
pip install -e stutter_detection/ExplainableVQA/DOVER
mkdir stutter_detection/ExplainableVQA/DOVER/pretrained_weights
wget https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth -P stutter_detection/ExplainableVQA/DOVER/pretrained_weights/

# Video quality assessment requirements
brew install ffmpeg
