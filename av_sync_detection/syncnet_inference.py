#!/usr/bin/python

import os, argparse, pickle, subprocess, glob
from shutil import rmtree
import argparse, subprocess, pickle, os, glob
from syncnet_python.SyncNetInstance import *
from syncnet_python.run_pipeline import track_shot, crop_video, inference_video, scene_detect

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry');
parser.add_argument('--videofile',      type=str, default='',   help='Input video file');
parser.add_argument('--reference',      type=str, default='',   help='Video reference');
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection');
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box');
parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration');
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped');
parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ========== ========== ========== ==========
# # EXECUTE INFERENCE
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========

if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
  rmtree(os.path.join(opt.work_dir,opt.reference))

if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
  rmtree(os.path.join(opt.crop_dir,opt.reference))

if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
  rmtree(os.path.join(opt.avi_dir,opt.reference))

if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
  rmtree(os.path.join(opt.frames_dir,opt.reference))

if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
  rmtree(os.path.join(opt.tmp_dir,opt.reference))

# ========== MAKE NEW DIRECTORIES ==========

os.makedirs(os.path.join(opt.work_dir,opt.reference))
os.makedirs(os.path.join(opt.crop_dir,opt.reference))
os.makedirs(os.path.join(opt.avi_dir,opt.reference))
os.makedirs(os.path.join(opt.frames_dir,opt.reference))
os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile,os.path.join(opt.avi_dir,opt.reference,'video.avi')))
output = subprocess.call(command, shell=True, stdout=None)

command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')))
output = subprocess.call(command, shell=True, stdout=None)

command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav')))
output = subprocess.call(command, shell=True, stdout=None)

# ========== FACE DETECTION ==========

faces = inference_video(opt)

# ========== SCENE DETECTION ==========

scene = scene_detect(opt)

# ========== FACE TRACKING ==========

alltracks = []
vidtracks = []

for shot in scene:

  if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
    alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num]))

# ========== FACE TRACK CROP ==========

for ii, track in enumerate(alltracks):
  vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

# ========== SAVE RESULTS ==========

savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

with open(savepath, 'wb') as fil:
  pickle.dump(vidtracks, fil)

rmtree(os.path.join(opt.tmp_dir,opt.reference))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
flist.sort()

# ==================== GET OFFSETS ====================

dists = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt,videofile=fname)
    dists.append(dist)
    print(f"\nPredicted time offset between the audio and the video: {offset} frames")
    print(f"Confidence probability of prediction: {conf:.4f} (0 = full confidence)")

# ==================== PRINT RESULTS TO FILE ====================

with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
    pickle.dump(dists, fil)
