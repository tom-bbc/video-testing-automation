#!/bin/bash
apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip3 install boto3
pip3 install awscli
pip3 install numpy
