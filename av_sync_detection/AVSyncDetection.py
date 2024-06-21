import os
import sys
import glob
import torch
import argparse
import torchvision
from datetime import datetime
from omegaconf import OmegaConf

sys.path.append('Synchformer/')
sys.path.append('Synchformer/model/modules/feat_extractors/visual/')

from Synchformer.dataset.dataset_utils import get_video_and_audio
from Synchformer.dataset.transforms import make_class_grid
from Synchformer.utils.utils import check_if_file_exists_else_download
from Synchformer.scripts.train_utils import get_model, get_transforms, prepare_inputs
from Synchformer.example import patch_config, decode_single_video_prediction, reencode_video, main


class AVSyncDetection():
    def __init__(self, device='cpu'):
        self.video_detection_results = []
        self.video_segment_index = 0

        self.offset_sec = 0.0
        self.v_start_i_sec = 0.0
        self.device = torch.device(device)

        self.vfps = 25
        self.afps = 16000
        self.in_size = 256

        # if the model does not exist try to download it from the server
        exp_name = '24-01-04T16-39-21'
        cfg_path = f'Synchformer/logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
        self.ckpt_path = f'Synchformer/logs/sync_models/{exp_name}/{exp_name}.pt'
        check_if_file_exists_else_download(cfg_path)
        check_if_file_exists_else_download(self.ckpt_path)

        # load config
        self.cfg = OmegaConf.load(cfg_path)

        # patch config
        self.cfg = patch_config(self.cfg)

    def process(self, directory_path, time_indexed_files=True):
        if os.path.isfile(directory_path):
            # Permits running on single input file
            if directory_path.endswith(".mp4"):
                segment_paths = [directory_path]
            else:
                exit(1)
        elif os.path.isdir(directory_path):
            # Gets list of AV files from local directory
            segment_paths = self.get_local_paths(dir=directory_path, time_indexed_files=time_indexed_files)
        else:
            exit(1)

        # Load the Syncformer model from checkpoint
        self.load_model()

        # Cycle through each AV file running detection algorithms
        for video_path in segment_paths:
            # Run video detection
            if time_indexed_files:
                timestamps = [datetime.strptime(f, '%H:%M:%S.%f') for f in video_path.split('/')[-1].replace('.mp4', '').split('_')[1:]]

            results = self.video_detection(video_path)
            print(f"\n * Full results: {results}")

            # Add local detection results to global results timeline (compensating for segment overlap)
            self.video_detection_results.append(results)
            self.video_segment_index += 1

        # # Plot global video detection results over all clips in timeline
        # global_start_time = datetime.strptime(video_segment_paths[0].split('/')[-1].replace('.mp4', '').split('_')[1], '%H:%M:%S.%f')
        # global_end_time = timestamps[-1]
        # print(f"Full timeline: {global_start_time.strftime('%H:%M:%S.%f')} => {global_end_time.strftime('%H:%M:%S.%f')}")

    def get_local_paths(self, dir, time_indexed_files=False):
        video_filenames = glob.glob(f"{dir}*.mp4")

        if time_indexed_files:
            sort_by_index = lambda path: int(path.split('/')[-1].split('_')[0][3:])
            video_filenames = list(sorted(video_filenames, key=sort_by_index))

        return video_filenames

    def load_model(self):
        # load the model
        _, self.model = get_model(self.cfg, self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    def video_detection(self, vid_path):
        print(f"\n * New video segment: {vid_path}", end='\n\n')

        # checking if the provided video has the correct frame rates
        print(f'Using video: {vid_path}')
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        if info['video_fps'] != self.vfps or info['audio_fps'] != self.afps or min(H, W) != self.in_size:
            print(f'Reencoding. vfps: {info["video_fps"]} -> {self.vfps};', end=' ')
            print(f'afps: {info["audio_fps"]} -> {self.afps};', end=' ')
            print(f'{(H, W)} -> min(H, W)={self.in_size}')
            vid_path = reencode_video(vid_path, self.vfps, self.afps, self.in_size)
        else:
            print(f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={self.in_size}')

        # load visual and audio streams
        # rgb: (Tv, 3, H, W) in [0, 225], audio: (Ta,) in [-1, 1]
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)

        # making an item (dict) to apply transformations
        item = dict(
            video=rgb, audio=audio, meta=meta, path=vid_path, split='test',
            targets={'v_start_i_sec': self.v_start_i_sec, 'offset_sec': self.offset_sec, },
        )

        # making the offset class grid similar to the one used in transforms
        max_off_sec = self.cfg.data.max_off_sec
        num_cls = self.cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
        if not (min(grid) <= item['targets']['offset_sec'] <= max(grid)):
            print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {grid}')

        # applying the test-time transform
        item = get_transforms(self.cfg, ['test'])['test'](item)

        # prepare inputs for inference
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, self.device)

        # forward pass
        with torch.set_grad_enabled(False):
            _, logits = self.model(
                vid.to(self.device, dtype=torch.float),
                aud.to(self.device, dtype=torch.float)
            )

        # simply prints the results of the prediction
        predictions = grid
        likelihoods = decode_single_video_prediction(logits, grid, item)

        return list(zip(
            [round(float(p), 1) for p in predictions],
            [round(float(l), 4) for l in likelihoods]
        ))


if __name__ == '__main__':
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='AVSyncDetection.py',
        description='Run Synchformer AV sync offset detection model over local AV segments.'
    )

    parser.add_argument('directory')
    parser.add_argument('-t', '--true-timestamps', action='store_true', default=False)
    parser.add_argument('-i', '--time-indexed-files', action='store_true', default=False)
    parser.add_argument('-d', '--device', default='mps')
    args = parser.parse_args()

    # Initialise and run AV sync model on input files
    detector = AVSyncDetection(args.device)
    detector.process(args.directory, time_indexed_files=args.time_indexed_files)
