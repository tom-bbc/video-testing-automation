# General Setup

* Firstly install external submodules: `git submodule update --init --recursive`
* Install requirements using pip: `pip install -r requirements.txt`
* Audio and video capture module is located within directory **capture**
* AV synchronisation detection module is located within directory **av_sync_detection**
* Stutter detection module is located within directory **stutter_detection**

<br>
<br>

# AV Capture System

* Setup mode to check input audio/video sources: `python capture/capture.py --setup-mode`
* Run capture pipeline to generate AV files: `python capture/capture.py -a AUDIO_SOURCE -v VIDEO_SOURCE --save-files`
* This capture audio and video in 10s segments and save them to the local directory **output/capture/**
* Halt capture by interrupting execution

#### General CLI

```
usage: capture.py [-h] [-s] [-na] [-nv] [-f] [-a AUDIO] [-v VIDEO]

Capture audio and video streams from a camera/microphone and split into segments for processing.

options:
  -h, --help            show this help message and exit
  -s, --setup-mode
  -na, --no-audio
  -nv, --no-video
  -f, --save-files
  -a AUDIO, --audio AUDIO
  -v VIDEO, --video VIDEO
```

<br>
<br>

# AV Synchronisation Detection

## Complete Detection System

* The complete build of the AV sync detection system uses Synchformer to predict AV offsets (as this was found to be the most accurate model during experimentation).
* Detection can be completed over a video file or directory of files.
* Can also enable **streaming** mode that continuously checks a directory for files and processes as they are added. This can be used in conjunction with the capture system to perform AV sync detection in real-time.
* Run inference on static files at **PATH**: `python AVSyncDetection.py PATH --plot`
* Run in streaming mode on captured video segments: `python AVSyncDetection.py ../output/capture/segments/ -stp`

<br>

## Experiments

### Synchformer

* Move inference script `synchformer_inference.py` into Synchformer submodule directory (and `cd` into this directory)
* Install requirements: `pip install omegaconf==2.0.6 av==10.0 einops timm==0.6.12`
* Run inference on MP4 file at **PATH**: `python synchformer_inference.py --vid_path PATH --device DEVICE`


### SparseSync

* Move inference script `sparsesync_inference.py` into SparseSync submodule directory (and `cd` into this directory)
* Install requirements: `pip install torch torchaudio torchvision omegaconf einops av`
* Run inference on MP4 file at **PATH**: `python sparsesync_inference.py --vid_path PATH --device DEVICE`


### SyncNet

#### Install

* Update the requirement `scenedetect` in file `requirements.txt` to the latest version using `scenedetect>=0.6.3`
* Then install requirements: `pip install -r requirements.txt`
* Download pre-trained SyncNet model by running `./download_model.sh`
* In file `SyncNetInstance.py`, remove instances of `.cuda()`
* In file `run_pipeline.py`, change device of face detection model by swapping line 187 to `DET = S3FD(device='cpu')`
* In file `detectors/s3fd/box_utils.py`, update depreciated `np.int` reference on line 38 to just `int`

#### Run

* Move inference script `syncnet_inference.py` into syncnet_python submodule directory (and `cd` into this directory)
* Run inference on MP4 file at **PATH**: `python syncnet_inference.py --videofile PATH`


### Vocalist

#### Install (for cpu)

* In `models/model.py` add `device` parameter to `init` method of `SyncTransformer` class.
* Pass the device parameter to all `TransformerEncoder` instances.
* In `models/transformer_encoder.py` add `device` parameter to `init` method of `TransformerEncoder` and `TransformerEncoderLayer` classes.
* Within the `TransformerEncoder` init method, pass the device parameter to all `TransformerEncoderLayer` instances.
* Within the `TransformerEncoderLayer` init method, add `self.device` as a field initialised from the input parameter.
* Add `self` to the inputs of the `buffered_future_mask` method of `TransformerEncoderLayer` and replace the inner `.cuda()` method call with `.to(self.device)`
* Ensure the attention mask is on the same device by adding `mask.to(self.device)` after line 153 of file `models/transformer_encoder.py`
* In `test_lrs2.py` add `data_root` as a `init` method parameter of the `Dataset` class, and pass this to the `get_image_list` method call.
* There is an issue with the attention mask, you must force i=the mask to dimension `16x5` by adding lines `dim1 = 5` and `dim2 = 16` within the `buffered_future_mask` of file `models/transformer_encoder.py`. You must then add a second try clause at line 120 within the file `models/multihead_attention.py` when the mask is used to try using the transpose through the statement `attn_weights += attn_mask.T.unsqueeze(0)`.

* `brew install cmake` and `pip install dlib`
* `pip install "librosa=0.9.1"`


* To use MPS device on Mac, must update permitted devices in file `Wav2Lip/face_detection/detection/core.py` by including `'mps' not in device` in if statement at line 27.

#### Run

* Wav2Lip preprocessing: `python wav2lip_preprocessing.py --results_dir prepared_data/putin-10s --input_videos ../../data/putin-10s.mp4`
* AV sync detection: `PYTORCH_ENABLE_MPS_FALLBACK=1 python vocalist_inference.py --input_data prepared_data/putin-10s`

<br>
<br>

# Stutter Detection

## Installing

### Installing Video Stutter Module
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

<br>

## Running

* Run inference on directory at **PATH**: `python StutterDetection.py PATH`
* This will timestamps of the any stuttering found in the audio or video files.

### General CLI

```
usage: StutterDetection.py [-h] [-f FRAMES] [-e EPOCHS] [-c] [-na] [-nv] [-t] directory

Run audio and video stutter detection algorithms over local AV segments.

positional arguments:
  directory

options:
  -h, --help            show this help message and exit
  -f FRAMES, --frames FRAMES
  -e EPOCHS, --epochs EPOCHS
  -c, --clean-video
  -na, --no-audio
  -nv, --no-video
  -t, --true-timestamps
```
