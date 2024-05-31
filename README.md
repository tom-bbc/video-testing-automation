# Stutter Detection

## Installing

### Installing Video Detection Module
1. Install ExplainableVQA deps:
```
git submodule update --init --recursive
pip install -r ExplainableVQA/requirements.txt
```
2. Install open_clip:

On Mac:
```
sed -i "" "92s/return x\[0\]/return x/" ExplainableVQA/open_clip/src/open_clip/modified_resnet.py
pip install -e ExplainableVQA/open_clip
```
On Linux:
```
sed -i '92s/return x\[0\]/return x/' ExplainableVQA/open_clip/src/open_clip/modified_resnet.py
pip install -e ExplainableVQA/open_clip
```
3. Install Dover:

On Mac first run this before continuing: `sed -i "" "4s/decord/eva-decord/" ExplainableVQA/DOVER/requirements.txt`
```
pip install -e ExplainableVQA/DOVER
mkdir ExplainableVQA/DOVER/pretrained_weights
wget https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth -P ExplainableVQA/DOVER/pretrained_weights/
```

## Running

### General CLI

```
usage: python main.py [-h] [-s] [-na] [-nv] [-a AUDIO] [-v VIDEO] [-f]

Capture audio and video streams from a camera/microphone and process detection algorithms over this content.

options:
  -h, --help            show this help message and exit
  -s, --setup-mode
  -na, --no-audio
  -nv, --no-video
  -f, --save-files
  -a AUDIO, --audio AUDIO
  -v VIDEO, --video VIDEO
```


# AV Synchronisation Detection

## SyncNet

### Install (for cpu)

* Update the requirement `scenedetect` in file `requirements.txt` to the latest version using `scenedetect>=0.6.3`
* Then install requirements: `pip install -r requirements.txt`
* Download pre-trained SyncNet model by running `./download_model.sh`
* In file `SyncNetInstance.py`, remove instances of `.cuda()`
* In file `run_pipeline.py`, change device of face detection model by swapping line 187 to `DET = S3FD(device='cpu')`
* In file `detectors/s3fd/box_utils.py`, update depreciated `np.int` reference on line 38 to just `int`

### Run

* Demo script: `python demo_syncnet.py --videofile data/example.avi --tmp_dir output`
* Using my inference script (after moving into syncnet_python dir): `python syncnet_inference.py --videofile ../data/putin-offset.mp4 --reference putin-offset --data_dir output`
