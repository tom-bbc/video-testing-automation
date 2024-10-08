# General Setup

* Requirements:
  * Clone external submodules: `git submodule update --init --recursive`
  * Set Python version to **3.10**: `pyenv global 3.10`
  * Install Python requirements using pip:
    * `python -m venv venv`
    * `source venv/bin/activate`
    * `pip install -r requirements.txt`
  * If on Mac, [download and install](https://github.com/matthutchinson/videosnap/releases) shell requirement **VideoSnap** (a macOS command line tool for recording video and audio from any attached capture device):
    * `wget https://github.com/matthutchinson/videosnap/releases/download/v0.0.9/videosnap-0.0.9.pkg`
    * `sudo installer -pkg videosnap-0.0.9.pkg -target /`
* Contents:
  * Audio and video capture module is located within directory **capture**
  * AV synchronisation detection using *Synchformer* is located within directory **av_sync_detection**
  * Stutter detection using *MaxVQA* and *Essentia* is located within directory **stutter_detection**
  * Video quality assessment using *Google UVQ* is located within directory **video_quality_assessment**

<br>

# AV Capture System

* Setup mode to check input audio/video sources: `python capture/capture.py --setup-mode`
* Run capture pipeline to generate AV files: `python capture/capture.py -a AUDIO_SOURCE -v VIDEO_SOURCE`
* This capture audio and video in 10s segments and save them to the local directory **output/capture/**
* Halt capture by interrupting execution with `CTRL+C`

#### General CLI

```
usage: capture.py [-h] [-m] [-na] [-nv] [-s] [-a AUDIO] [-v VIDEO] [-o OUTPUT_PATH]

Capture audio and video streams from a camera/microphone and split into segments for processing.

options:
  -h, --help            show this help message and exit
  -m, --setup-mode      display video to be captured in setup mode with no capture/processing
  -na, --no-audio       do not include audio in captured segments
  -nv, --no-video       do not include video in captured segments
  -s, --split-av-out    output audio and video in separate files (WAV and MP4)
  -a AUDIO, --audio AUDIO
                        index of input audio device
  -v VIDEO, --video VIDEO
                        index of input video device
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        directory to output captured video segments to
```

<br>

# AV Synchronisation Detection

## Complete Detection System

* The complete build of the AV sync detection system uses Synchformer to predict AV offsets (as this was found to be the most accurate model during experimentation).
* Detection can be completed over a video file or directory of files.
* Can also enable **streaming** mode that continuously checks a directory for files and processes as they are added. This can be used in conjunction with the capture system to perform AV sync detection in real-time.
* Run inference on static files at **PATH**: `python AVSyncDetection.py PATH --plot`
* Run in streaming mode on captured video segments: `python AVSyncDetection.py ../output/capture/segments/ -sip`
* If running on an Apple Silicon Mac: `python AVSyncDetection.py PATH -p --device mps`
* If running on a GPU: `python AVSyncDetection.py PATH -p --device cuda`

#### General CLI

```
usage: AVSyncDetection.py [-h] [-p] [-s] [-i] [-d DEVICE] [-t TRUE_OFFSET] directory

Run Synchformer AV sync offset detection model over local AV segments.

positional arguments:
  directory

options:
  -h, --help            show this help message and exit
  -p, --plot            plot sync predictions as generated by model
  -s, --streaming       real-time detection of streamed input by continuously locating & processing video segments
  -i, --time-indexed-files
                        label output predictions with available timestamps of input video segments
  -d DEVICE, --device DEVICE
                        harware device to run model on
  -t TRUE_OFFSET, --true-offset TRUE_OFFSET
                        known true av offset of the input video
```


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


## Running

* Run inference on directory or video/audio file at **PATH**: `python StutterDetection.py PATH`
* This will output a plot of the "motion fluency" over the course of the video (low fluency may indicate stuttering events) and/or a plot of audio stutter times detected in the waveform.

### General CLI

```
usage: StutterDetection.py [-h] [-na] [-nv] [-c] [-t] [-i] [-f FRAMES] [-e EPOCHS]
                           [-d DEVICE]
                           directory

Run audio and video stutter detection algorithms over local AV segments.

positional arguments:
  directory

options:
  -h, --help            show this help message and exit
  -na, --no-audio       Do not perform stutter detection on the audio track
  -nv, --no-video       Do not perform stutter detection on the video track
  -c, --clean-video     Testing on clean stutter-free videos (for experimentation)
  -t, --true-timestamps
                        Plot known stutter times on the output graph, specified in
                        'true-stutter-timestamps.json
  -i, --time-indexed-files
                        Label batch of detections over video segments with their
                        time range (from filename)
  -f FRAMES, --frames FRAMES
                        Number of frames to downsample video to
  -e EPOCHS, --epochs EPOCHS
                        Number of times to repeat inference per video
  -d DEVICE, --device DEVICE
                        Specify processing hardware
```
