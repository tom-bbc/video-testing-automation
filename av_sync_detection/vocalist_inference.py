# Wav2Lip imports
from os import path
import numpy as np
import cv2, os, sys, argparse
import subprocess
from tqdm import tqdm
import torch
from pathlib import Path

sys.path.append('Wav2Lip/')

from Wav2Lip.audio import load_wav, melspectrogram
from Wav2Lip.face_detection import FaceAlignment, LandmarksType
from Wav2Lip.models import Wav2Lip

# Vocalist imports
import torch
from models.model import SyncTransformer
from torch.utils import data as data_utils
from tqdm import tqdm
from os.path import dirname, join, basename, isfile
import soundfile as sf
import math
import os
from natsort import natsorted
from glob import glob
import cv2
import numpy as np
from torchaudio.transforms import MelScale
import argparse
from hparams import hparams

# Wav2Lip global vars
img_size = 96
max_frame_res = 720
face_det_batch_size = 16
face_res = 180
min_frame_res = 480
pads = [0, 10, 0, 0]
wav2lip_batch_size = 128
mel_step_size = 16

# Wav2Lip functions
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def rescale_frames(images, detector):
    rect = detector.get_detections_for_batch(np.array([images[0]]))[0]
    if rect is None:
        raise ValueError('Face not detected!')
    h, w = images[0].shape[:-1]

    x1, y1, x2, y2 = rect

    face_size = max(np.abs(y1 - y2), np.abs(x1 - x2))

    diff = np.abs(face_size - face_res)
    for factor in range(2, 16):
        downsampled_res = face_size // factor
        if min(h//factor, w//factor) < min_frame_res: break
        if np.abs(downsampled_res - face_res) >= diff: break

    factor -= 1
    if factor == 1: return images

    return [cv2.resize(im, (im.shape[1]//(factor), im.shape[0]//(factor))) for im in images]


def face_detect(images, device='cpu'):
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    batch_size = face_det_batch_size
    images = rescale_frames(images, detector)

    while 1:
        predictions = []
        try:
            with tqdm(range(0, len(images), batch_size)) as I:
                I.set_description("Face detection prediction")
                for i in I:
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    print("Face prediction complete.")

    results = []
    pady1, pady2, padx1, padx2 = pads
    with tqdm(zip(predictions, images)) as P:
        P.set_description("Bounding box prediction")
        for rect, image in P:
            if rect is None:
                raise ValueError('Face not detected!')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])


    boxes = get_smoothened_boxes(np.array(results), T=5)

    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results, images

def datagen(frames, face_det_results, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		if i >= len(frames): raise ValueError('Equal or less lengths only')

		frame_to_save = frames[i].copy()
		face, coords, valid_frame = face_det_results[i].copy()
		if not valid_frame:
			print("Invalid frame detected.")
			continue

		face = cv2.resize(face, (img_size, img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def increase_frames(frames, l):
	## evenly duplicating frames to increase length of video
	while len(frames) < l:
		dup_every = float(l) / len(frames)

		final_frames = []
		next_duplicate = 0.

		for i, f in enumerate(frames):
			final_frames.append(f)

			if int(np.ceil(next_duplicate)) == i:
				final_frames.append(f)

			next_duplicate += dup_every

		frames = final_frames

	return frames[:l]


def load_model(path, device):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = torch.load(path, map_location=torch.device(device))
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		new_s[k.replace('module.', '')] = v

	model.load_state_dict(new_s)
	model = model.to(device)

	return model.eval()

# Main Wav2Lip module
def wav2lip_preprocess(video, results_dir, device='cpu', model_checkpoint='Wav2Lip/models/wav2lip.pth'):
    model = load_model(model_checkpoint, device)

    if not os.path.isdir(results_dir): os.makedirs(results_dir)
    assert os.path.isfile(video)

    audio_src = video
    temp_audio = os.path.join(results_dir, 'audio.wav')

    command = f'ffmpeg -y -i {audio_src} -ac 1 -ar 16000 {temp_audio}'
    print(f"FFMPEG command: {command}")
    subprocess.call(command, shell=True)
    print(f"Audio signal separated into file: {temp_audio}")

    wav = load_wav(temp_audio, 16000)
    mel = melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan!')

    video_stream = cv2.VideoCapture(video)

    fps = video_stream.get(cv2.CAP_PROP_FPS)
    mel_idx_multiplier = 80./fps

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break

        if min(frame.shape[:-1]) > max_frame_res:
            h, w = frame.shape[:-1]
            scale_factor = min(h, w) / float(max_frame_res)
            h = int(h/scale_factor)
            w = int(w/scale_factor)

            frame = cv2.resize(frame, (w, h))
        full_frames.append(frame)

    mel_chunks = []
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    if len(full_frames) < len(mel_chunks):
        raise ValueError('#Frames, audio length mismatch')
    elif len(full_frames) != len(mel_chunks):
        full_frames = full_frames[:len(mel_chunks)]

    face_det_results, full_frames = face_detect(full_frames.copy())

    gen = datagen(full_frames.copy(), face_det_results, mel_chunks)

    with tqdm(enumerate(gen)) as G:
        G.set_description("Saving face frames")
        for i, (img_batch, mel_batch, frames, coords) in G:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for i, (pl, f, c) in enumerate(zip(pred, frames, coords)):
                y1, y2, x1, x2 = c
                pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = pl
                output_path = path.join(results_dir, '{}.jpg'.format(i))
                cv2.imwrite(output_path, pl)


# Vocalist global variables
v_context = 5
mel_step_size = 16  # num_audio_elements/hop_size
BATCH_SIZE = 1
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, f_max=hparams.fmax,
                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney')


# Vocalist functions

class Dataset(object):
    def __init__(self, video_paths, device='cpu'):
        self.all_videos = video_paths
        self.device = device

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath):
        return sf.read(wavpath)[0]

    def get_window(self, start_frame, end):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, end):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            vidname = self.all_videos[idx]
            wavpath = join(vidname, "audio.wav")

            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            wav = self.get_wav(wavpath)
            min_length = min(len(img_names), math.floor(len(wav) / 640))
            lastframe = min_length - v_context

            img_name = os.path.join(vidname, '0.jpg')
            window_fnames = self.get_window(img_name, len(img_names))
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue
            # H, W, T, 3 --> T*3
            vid = np.concatenate(window, axis=2) / 255.
            vid = vid.transpose(2, 0, 1)
            vid = torch.FloatTensor(vid[:, 48:])

            aud_tensor = torch.FloatTensor(wav)
            spec = torch.stft(aud_tensor, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            # NORMALIZED MEL
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            mels = normalized_mel.unsqueeze(0)
            if torch.any(torch.isnan(vid)) or torch.any(torch.isnan(mels)):
                continue
            if vid==None or mels==None:
                continue
            return vid, mels, lastframe


def calc_pdist(model, feat1, feat2, vshift=15, device='cpu'):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2.permute(1,2,3,0).contiguous(), (vshift,vshift)).permute(3,0,1,2).contiguous()

    dists = []
    num_rows_dist = len(feat1)
    with tqdm(range(0, num_rows_dist)) as R:
        R.set_description(f"Passing video frames through model ({device})")
        for i in R:
            raw_sync_scores = model(feat1[i].unsqueeze(0).repeat(win_size, 1, 1, 1).to(device), feat2p[i:i + win_size, :].to(device))
            dist_measures = raw_sync_scores.clone().cpu()
            if i in range(vshift):
                dist_measures[0:vshift-i] = torch.tensor(-1000, dtype=torch.float).to(device)
            elif i in range(num_rows_dist-vshift,num_rows_dist):
                dist_measures[vshift+num_rows_dist-i:] = torch.tensor(-1000, dtype=torch.float).to(device)

            dists.append(dist_measures)

    return dists


def eval_model(test_data_loader, device, model, no_preds_display=5):
    prog_bar = tqdm(enumerate(test_data_loader))
    samplewise_acc_k5, samplewise_acc_k7, samplewise_acc_k9, samplewise_acc_k11, samplewise_acc_k13, samplewise_acc_k15 = [],[],[],[],[],[]
    for step, (vid, aud, lastframe) in prog_bar:
        model.eval()
        with torch.no_grad():
            vid = vid.view(BATCH_SIZE,(lastframe+v_context),3,48,96)
            batch_size = 20
            lastframe = lastframe.item()
            lim_in = []
            lcc_in = []
            for i in range(0, lastframe, batch_size):
                im_batch = [vid[:, vframe:vframe + v_context, :, :, :].view(BATCH_SIZE, -1, 48, 96) for vframe in
                            range(i, min(lastframe, i + batch_size))]
                im_in = torch.cat(im_batch, 0)
                lim_in.append(im_in)

                cc_batch = [aud[:, :, :, int(80.*(vframe/float(hparams.fps))):int(80.*(vframe/float(hparams.fps)))+mel_step_size] for vframe in
                            range(i, min(lastframe, i + batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                lcc_in.append(cc_in)

            lim_in = torch.cat(lim_in, 0)
            lcc_in = torch.cat(lcc_in, 0)
            dists = calc_pdist(model, lim_in, lcc_in, vshift=hparams.v_shift, device=device)

            # K=5
            dist_tensor_k5 = torch.stack(dists)
            offsets_k5 = hparams.v_shift - torch.argmax(dist_tensor_k5, dim=1)

            # K=7
            dist_tensor_k7 = (dist_tensor_k5[1:-1] + dist_tensor_k5[2:] + dist_tensor_k5[:-2]) / 3  # inappropriate to average over 0,0,20 for example
            dk7_m1 = torch.mean(dist_tensor_k5[:2], dim=0).unsqueeze(0)
            dk7_p1 = torch.mean(dist_tensor_k5[-2:], dim=0).unsqueeze(0)
            dist_tensor_k7 = torch.cat([dk7_m1, dist_tensor_k7, dk7_p1], dim=0)
            offsets_k7 = hparams.v_shift - torch.argmax(dist_tensor_k7, dim=1)

            # K=9
            dist_tensor_k9 = (dist_tensor_k5[2:-2] + dist_tensor_k5[1:-3] + dist_tensor_k5[3:-1] + dist_tensor_k5[:-4] + dist_tensor_k5[4:]) / 5
            dk9_m1 = torch.mean(dist_tensor_k5[:4], dim=0).unsqueeze(0)
            dk9_p1 = torch.mean(dist_tensor_k5[-4:], dim=0).unsqueeze(0)
            dk9_m2 = torch.mean(dist_tensor_k5[:3], dim=0).unsqueeze(0)
            dk9_p2 = torch.mean(dist_tensor_k5[-3:], dim=0).unsqueeze(0)
            dist_tensor_k9 = torch.cat([dk9_m2, dk9_m1, dist_tensor_k9, dk9_p1, dk9_p2], dim=0)
            offsets_k9 = hparams.v_shift - torch.argmax(dist_tensor_k9, dim=1)

            # K=11
            dist_tensor_k11 = (dist_tensor_k5[3:-3] + dist_tensor_k5[2:-4] + dist_tensor_k5[4:-2] +
                               dist_tensor_k5[1:-5] + dist_tensor_k5[5:-1] + dist_tensor_k5[:-6] + dist_tensor_k5[6:]) / 7
            dk11_m1 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk11_p1 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)
            dk11_m2 = torch.mean(dist_tensor_k5[:5], dim=0).unsqueeze(0)
            dk11_p2 = torch.mean(dist_tensor_k5[-5:], dim=0).unsqueeze(0)
            dk11_m3 = torch.mean(dist_tensor_k5[:4], dim=0).unsqueeze(0)
            dk11_p3 = torch.mean(dist_tensor_k5[-4:], dim=0).unsqueeze(0)
            dist_tensor_k11 = torch.cat([dk11_m3, dk11_m2, dk11_m1, dist_tensor_k11, dk11_p1, dk11_p2, dk11_p3], dim=0)
            offsets_k11 = hparams.v_shift - torch.argmax(dist_tensor_k11, dim=1)

            # K=13
            dist_tensor_k13 = (dist_tensor_k5[4:-4] + dist_tensor_k5[3:-5] + dist_tensor_k5[5:-3] +
                               dist_tensor_k5[2:-6] + dist_tensor_k5[6:-2] + dist_tensor_k5[1:-7] +
                               dist_tensor_k5[7:-1] + dist_tensor_k5[:-8] + dist_tensor_k5[8:]) / 9
            dk13_m1 = torch.mean(dist_tensor_k5[:8], dim=0).unsqueeze(0)
            dk13_p1 = torch.mean(dist_tensor_k5[-8:], dim=0).unsqueeze(0)
            dk13_m2 = torch.mean(dist_tensor_k5[:7], dim=0).unsqueeze(0)
            dk13_p2 = torch.mean(dist_tensor_k5[-7:], dim=0).unsqueeze(0)
            dk13_m3 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk13_p3 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)
            dk13_m4 = torch.mean(dist_tensor_k5[:5], dim=0).unsqueeze(0)
            dk13_p4 = torch.mean(dist_tensor_k5[-5:], dim=0).unsqueeze(0)

            dist_tensor_k13 = torch.cat([dk13_m4, dk13_m3, dk13_m2, dk13_m1, dist_tensor_k13, dk13_p1, dk13_p2, dk13_p3, dk13_p4], dim=0)
            offsets_k13 = hparams.v_shift - torch.argmax(dist_tensor_k13, dim=1)

            # K=15
            dist_tensor_k15 = (dist_tensor_k5[5:-5] + dist_tensor_k5[4:-6] + dist_tensor_k5[6:-4] +
                               dist_tensor_k5[3:-7] + dist_tensor_k5[7:-3] + dist_tensor_k5[2:-8] +
                               dist_tensor_k5[8:-2] + dist_tensor_k5[1:-9] + dist_tensor_k5[9:-1] +
                               dist_tensor_k5[:-10] + dist_tensor_k5[10:]) / 11
            dk15_m1 = torch.mean(dist_tensor_k5[:10], dim=0).unsqueeze(0)
            dk15_p1 = torch.mean(dist_tensor_k5[-10:], dim=0).unsqueeze(0)
            dk15_m2 = torch.mean(dist_tensor_k5[:9], dim=0).unsqueeze(0)
            dk15_p2 = torch.mean(dist_tensor_k5[-9:], dim=0).unsqueeze(0)
            dk15_m3 = torch.mean(dist_tensor_k5[:8], dim=0).unsqueeze(0)
            dk15_p3 = torch.mean(dist_tensor_k5[-8:], dim=0).unsqueeze(0)
            dk15_m4 = torch.mean(dist_tensor_k5[:7], dim=0).unsqueeze(0)
            dk15_p4 = torch.mean(dist_tensor_k5[-7:], dim=0).unsqueeze(0)
            dk15_m5 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk15_p5 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)

            dist_tensor_k15 = torch.cat([dk15_m5, dk15_m4, dk15_m3, dk15_m2, dk15_m1, dist_tensor_k15, dk15_p1, dk15_p2, dk15_p3, dk15_p4, dk15_p5], dim=0)
            offsets_k15 = hparams.v_shift - torch.argmax(dist_tensor_k15, dim=1)

            # predictions, counts = torch.unique(offsets_k5, return_counts=True)
            # print(f"\nPredicted offsets (K05): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            # predictions, counts = torch.unique(offsets_k7, return_counts=True)
            # print(f"Predicted offsets (K07): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            # predictions, counts = torch.unique(offsets_k9, return_counts=True)
            # print(f"Predicted offsets (K09): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            # predictions, counts = torch.unique(offsets_k11, return_counts=True)
            # print(f"Predicted offsets (K11): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            # predictions, counts = torch.unique(offsets_k13, return_counts=True)
            # print(f"Predicted offsets (K13): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            # predictions, counts = torch.unique(offsets_k15, return_counts=True)
            # print(f"Predicted offsets (K15): {[f'{p}f: {100*(c/counts.sum()):.1f}%' for p, c in zip(predictions, counts)]}")

            all_offsets = torch.cat((offsets_k5, offsets_k7, offsets_k9, offsets_k11, offsets_k13, offsets_k15), 0)
            predictions, counts = torch.unique(all_offsets, return_counts=True)
            likelihoods = [100 * (c / counts.sum()) for c in counts]
            predictions_by_likelihood = sorted(list(zip(predictions, likelihoods)), key=lambda pl: pl[1], reverse=True)
            no_preds_display = min(len(predictions), no_preds_display)

            print("\nPredictions:")
            for p, l in predictions_by_likelihood[:no_preds_display]: print(f" * Delay: {p} fr, {p/25:.2f} s --- Likelihood: {l:.1f}%")


# Main Vocalist module
def vocalist_process(input_data_dir, device='cpu', model_checkpoint="vocalist_weights/vocalist_5f_lrs2.pth"):
    # Params
    BATCH_SIZE = 1
    input_data_dirs = [input_data_dir]

    # Model
    checkpoint = torch.load(model_checkpoint, map_location=torch.device(device))
    model = SyncTransformer(device=device).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Dataset and Dataloader setup
    print(f"Input data directory: {input_data_dirs}")
    inf_data = Dataset(input_data_dirs, device)
    inf_data_loader = data_utils.DataLoader(inf_data, batch_size=BATCH_SIZE, num_workers=0)

    eval_model(inf_data_loader, device, model)


# Full processing pipleine
if __name__ == '__main__':
    # Parse input parameters
    parser = argparse.ArgumentParser(description='Extract audio and speaker frames from an input video.')
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--tmp_data_dir', type=str, help='Folder to save segmented speaker frames and audio to.', default='./tmp')
    parser.add_argument('--wav2lip_model', type=str, help='Location of Wav2Lip model checkpoint file.', default='Wav2Lip/models/wav2lip.pth')
    parser.add_argument('--vocalist_model', type=str, help='Location of Vocalist model checkpoint file.', default='vocalist_weights/vocalist_5f_lrs2.pth')
    parser.add_argument('--no_preprocess', action='store_true', default=False)
    parser.add_argument('--no_detection', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    # Check input file exists and create temporary directory
    video_id = Path(args.input_video).stem
    temp_dir = os.path.join(args.tmp_data_dir, video_id)

    if not os.path.isfile(args.input_video) and not os.path.isdir(temp_dir):
        raise FileNotFoundError(f"Input video file '{args.input_video}' does not exist.")

    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    if not args.no_preprocess:
        print("Running pre-processing step...", end='\n\n')
        wav2lip_preprocess(args.input_video, temp_dir, device=args.device, model_checkpoint=args.wav2lip_model)
        print("\nPre-processing complete.", end='\n\n')

    # Run av sync detection
    if not args.no_detection:
        print("Running AV delay prediction step...", end='\n\n')
        vocalist_process(temp_dir, device=args.device, model_checkpoint=args.vocalist_model)
        print("\nAV delay prediction complete.", end='\n\n')
