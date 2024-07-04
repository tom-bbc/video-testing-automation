import os
import sys
import math
import time
import glob
import torch
import pathlib
import argparse
import numpy as np
import torchvision
import cmasher as cmr
from datetime import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

sys.path.append('Synchformer/')
sys.path.append('Synchformer/model/modules/feat_extractors/visual/')

from Synchformer.dataset.dataset_utils import get_video_and_audio
from Synchformer.dataset.transforms import make_class_grid
from Synchformer.utils.utils import check_if_file_exists_else_download
from Synchformer.scripts.train_utils import get_model, get_transforms, prepare_inputs
from Synchformer.example import patch_config, decode_single_video_prediction, reencode_video


class AVSyncDetection():
    def __init__(self, device='cpu', true_offset=None):
        self.video_detection_results = {}
        self.video_segment_index = 0

        self.offset_sec = 0.0
        self.v_start_i_sec = 0.0
        self.device = torch.device(device)

        if true_offset is not None:
            self.true_offset = float(true_offset)

        self.vfps = 25
        self.afps = 16000
        self.in_size = 256

        self.output_file = "av_sync_predictions.txt"

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

        self.system_timeout = 30
        self.retry_wait_time = 10

    def continuous_processing(self, directory_path, time_indexed_files=False, output_to_file=True, plot=True):
        # Setup
        if output_to_file:
            output_file = os.path.join(directory_path, self.output_file)
            with open(output_file, 'a') as file:
                file.write("\n--------------------------------------------------------------------------------\n")

        # Only allow continuous processing on directories
        if not os.path.isdir(directory_path):
            exit(1)

        # Load the Syncformer model from checkpoint
        self.load_model()

        # Gets list of AV files from local directory
        segment_file_paths = self.get_local_paths(dir=directory_path, time_indexed_files=time_indexed_files)
        processed_files = []
        print(f"New files found: {segment_file_paths}")

        while True:
            if len(segment_file_paths) > 0:
                video_path = segment_file_paths[0]

                if len(segment_file_paths) == 1: time.sleep(self.retry_wait_time // 2)
                if not os.access(video_path, os.R_OK): time.sleep(self.retry_wait_time)

                predictions = self.video_detection(video_path)
                video_id = pathlib.Path(video_path).stem
                self.video_detection_results.update({video_id: predictions})
                processed_files.append(video_path)

                if output_to_file:
                    with open(output_file, 'a') as file:
                        file.writelines([
                            f"\nInput video: {video_id}",
                            f"\nPredictions: {self.get_top_preds(predictions)}\n"
                        ])

            if len(segment_file_paths) > 1:
                segment_file_paths = segment_file_paths[1:]
            else:
                print("\nChecking for new files.")
                new_files = self.get_local_paths(dir=directory_path, time_indexed_files=time_indexed_files)
                segment_file_paths = [f for f in new_files if f not in processed_files]

                if len(segment_file_paths) == 0:
                    retry_attempt = 0
                    while len(segment_file_paths) == 0:
                        if retry_attempt >= self.system_timeout // self.retry_wait_time:
                            break

                        print(f"No new files located. Retry attempt: {retry_attempt + 1} / {self.system_timeout // self.retry_wait_time}")
                        retry_attempt += 1
                        time.sleep(self.retry_wait_time)

                        new_files = self.get_local_paths(dir=directory_path, time_indexed_files=time_indexed_files)
                        segment_file_paths = [f for f in new_files if f not in processed_files]

                    if len(segment_file_paths) == 0:
                        print("Shutting down processing.")
                        break

                print(f"New files found: {segment_file_paths}")

            if plot:
                self.plot(directory_path, time_indexed_files)

    def process(self, directory_path, time_indexed_files=False, output_to_file=True, plot=True):
        # Setup
        if output_to_file:
            output_file = os.path.join(directory_path, self.output_file)
            with open(output_file, 'a') as file:
                file.write("\n--------------------------------------------------------------------------------\n")

        if os.path.isfile(directory_path):
            # Permits running on single input file
            if directory_path.endswith(".mp4"):
                segment_paths = [directory_path]
                directory_path = os.path.dirname(directory_path)
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
            predictions = self.video_detection(video_path)
            video_id = pathlib.Path(video_path).stem
            self.video_detection_results.update({video_id: predictions})

            # Add local detection results to global results timeline (compensating for segment overlap)
            self.video_segment_index += 1

            if output_to_file:
                with open(output_file, 'a') as file:
                    file.writelines([
                        f"\nInput video: {video_id}",
                        f"\nPredictions: {self.get_top_preds(predictions)}\n"
                    ])

            if plot:
                self.plot(directory_path, time_indexed_files)

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
        print(f"\n--------------------------------------------------------------------------------\n")

        # Check file exists & is accessible
        if not os.path.isfile(vid_path) or not os.access(vid_path, os.R_OK):
            time.sleep(self.retry_wait_time // 2)

        # checking if the provided video has the correct frame rates
        print(f'Using video: {vid_path}')
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        if 'video_fps' not in info or 'audio_fps' not in info or info['video_fps'] != self.vfps or info['audio_fps'] != self.afps or min(H, W) != self.in_size:
            vid_path = reencode_video(vid_path, self.vfps, self.afps, self.in_size)
        else:
            print(f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={self.in_size}')

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
        likelihoods = decode_single_video_prediction(logits, grid, item)

        return list(zip(
            [round(float(pred), 1) for pred in grid],
            [round(float(prob), 4) for prob in likelihoods]
        ))

    @staticmethod
    def get_top_preds(preds_by_prob, threshold=0.001, num_return_preds=10):
        preds_by_prob = filter(lambda pred_and_prob: pred_and_prob[-1] > threshold, preds_by_prob)
        sorted_preds = list(sorted(preds_by_prob, key=lambda pred_and_prob: pred_and_prob[-1], reverse=True))
        top_predictions = sorted_preds[:min(num_return_preds, len(sorted_preds))]
        return top_predictions

    @staticmethod
    def narrow_pred_range(preds_by_prob, new_range_bound=1):
        filter_function = lambda pred_and_prob: (-new_range_bound <= pred_and_prob[0]) and (pred_and_prob[0] <= new_range_bound)
        preds_by_prob = list(filter(filter_function, preds_by_prob))
        return preds_by_prob

    def plot(self, output_dir='./', time_indexed_files=False):
        # Plot global video detection results over all clips in timeline
        plt.style.use('seaborn-v0_8')

        if len(self.video_detection_results) == 0:
            return

        x_axis_vals = []
        x_axis_labels = []
        y_axis = []
        colour_by_prob = []

        for video_index, (video_id, prediction) in enumerate(self.video_detection_results.items()):
            prediction = self.narrow_pred_range(prediction)

            if time_indexed_files:
                times = (
                    datetime.strptime(video_id.split('_')[1], '%H:%M:%S.%f'),
                    datetime.strptime(video_id.split('_')[2], '%H:%M:%S.%f')
                )

                x_value = f"     {datetime.strftime(times[0], '%H:%M:%S')} \n-> {datetime.strftime(times[1], '%H:%M:%S')}"
            else:
                x_value = video_id

            for pred, prob in prediction:
                x_axis_vals.append(video_index)
                x_axis_labels.append(x_value)
                y_axis.append(pred)
                colour_by_prob.append(prob)

        plot_width = max(math.ceil(len(np.unique(x_axis_labels)) * 0.7), 13)
        offset_step = 0.2
        y_limit = round(round(np.max(np.absolute(y_axis)) / offset_step) * offset_step + offset_step, 1)

        if y_limit > 1.8:
            point_size = 400
        elif y_limit > 1.4:
            point_size = 600
        else:
            point_size = 800

        fig, ax = plt.subplots(1, 1, figsize=(plot_width, 9))
        colour_map = cmr.get_sub_cmap('Greens', start=np.min(colour_by_prob), stop=np.max(colour_by_prob))
        predictions_plot = ax.scatter(x_axis_vals, y_axis, c=colour_by_prob, cmap=colour_map, s=point_size, zorder=10)

        for video_index, (video_id, prediction) in enumerate(self.video_detection_results.items()):
            max_likelihood_idx = np.argmax([prob for pred, prob in prediction])
            max_likelihood_prediction = prediction[max_likelihood_idx][0]
            ax.scatter(float(video_index), float(max_likelihood_prediction), s=point_size, facecolors='none', edgecolors='k', linewidth=5)

        if self.true_offset is not None:
            plt.axhline(y=self.true_offset, linestyle='-', c='k', linewidth=4, label='True Offset')

        plt.xticks(fontsize='small', rotation=90)
        ax.set_xticks(x_axis_vals)
        ax.set_xticklabels(x_axis_labels)

        ax.set_yticks(np.arange(-y_limit + offset_step, y_limit, offset_step))
        plt.yticks(fontsize='x-large')

        ax.set_xlabel("Video segment", fontsize='xx-large')
        ax.set_ylabel("Predicted Offset (s)", fontsize='xx-large')

        ax.set_title(f"Predicted AV Offset per Video Segment\n", fontsize=20)
        ax.grid(which='major', linewidth=1, zorder=0)

        cbar = fig.colorbar(predictions_plot, ax=ax, orientation='vertical', extend='both', ticks=np.arange(0, 1.1, 0.1), fraction=0.03)
        cbar.set_label(label='Likelihood', fontsize='xx-large')
        cbar.ax.tick_params(labelsize='x-large')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'av_sync_plot.png'))
        print(f"\nPredictions plot generated: {os.path.join(output_dir, 'av_sync_plot.png')}")
        plt.close()


if __name__ == '__main__':
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='AVSyncDetection.py',
        description='Run Synchformer AV sync offset detection model over local AV segments.'
    )

    parser.add_argument('directory')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help="plot sync predictions as generated by model")
    parser.add_argument('-s', '--streaming', action='store_true', default=False, help="real-time detection of streamed input by continuously locating & processing video segments")
    parser.add_argument('-i', '--time-indexed-files', action='store_true', default=False, help="label output predictions with available timestamps of input video segments")
    parser.add_argument('-d', '--device', default='cpu', help="harware device to run model on")
    parser.add_argument('-t', '--true-offset', default=None, help="known true av offset of the input video")

    args = parser.parse_args()

    # Initialise and run AV sync model on input files
    detector = AVSyncDetection(args.device, args.true_offset)

    if args.streaming:
        detector.continuous_processing(
            args.directory,
            time_indexed_files=args.time_indexed_files,
            plot=args.plot
        )
    else:
        detector.process(
            args.directory,
            time_indexed_files=args.time_indexed_files,
            plot=args.plot
        )