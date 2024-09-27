import os
import cv2
import json
import math
import glob
import pathlib
import argparse
import numpy as np
from scipy.io import wavfile
from datetime import datetime
from time import time as timer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle

from EssentiaAudioDetector import AudioDetector
from MaxVQAVideoDetector import VideoDetector

Object = lambda **kwargs: type('Object', (), kwargs)


class StutterDetection():
    def __init__(self, video_downsample_frames=64, audio_fps=44100, device='cpu'):
        self.audio_detector = AudioDetector()
        self.video_detector = VideoDetector(frames=video_downsample_frames, device=device)
        self.audio_detection_results = []
        self.video_detection_results = np.array([[]]*16)
        self.audio_fps = audio_fps
        self.audio_segment_index = 0
        self.video_segment_index = 0

    def process(self, directory_path, truth=None, audio_detection=True, video_detection=True, plot=True, time_indexed_files=True, inference_epochs=1, output_directory='./'):
        if os.path.isfile(directory_path):
            # Permits running on single input file
            if directory_path.endswith(".mp4"):
                video_segment_paths = [directory_path]
                audio_segment_paths = []
            elif directory_path.endswith(".wav"):
                audio_segment_paths = [directory_path]
                video_segment_paths = []
            else:
                exit(1)
        elif os.path.isdir(directory_path):
            # Gets list of AV files from local directory
            audio_segment_paths, video_segment_paths = self.get_local_paths(directory_path, audio_detection, video_detection, time_indexed_files)
        else:
            exit(1)

        # Cycle through each AV file running detection algorithms
        for index in range(max(len(audio_segment_paths), len(video_segment_paths))):
            # Run audio detection
            if audio_detection and index < len(audio_segment_paths):
                audio_path = audio_segment_paths[index]
                audio_segment = self.get_local_audio(audio_path)
                print(f"New audio segment: {audio_path.split('/')[-1]} {audio_segment.shape}")

                if time_indexed_files:
                    timestamps = [datetime.strptime(f, '%H:%M:%S.%f') for f in audio_path.split('/')[-1].replace('.wav', '').split('_')[1:]]

                    results = self.audio_detection(
                        audio_segment,
                        plot=plot,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        output_dir=output_directory
                    )
                else:
                    results = self.audio_detection(
                        audio_segment,
                        plot=plot,
                        output_dir=output_directory
                    )

                self.audio_segment_index += 1

            # Run video detection
            if video_detection and index < len(video_segment_paths):
                video_path = video_segment_paths[index]
                video_segment = self.get_local_video(video_path)
                print(f"New video segment: {video_path.split('/')[-1]} {video_segment.shape}")

                if time_indexed_files:
                    timestamps = [datetime.strptime(f, '%H:%M:%S.%f') for f in video_path.split('/')[-1].replace('.mp4', '').split('_')[1:]]

                    results = self.video_detection(
                        video_segment,
                        plot=plot,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        epochs=inference_epochs,
                        output_dir=output_directory
                    )
                else:
                    results = self.video_detection(
                        video_segment,
                        plot=plot,
                        epochs=inference_epochs,
                        output_dir=output_directory
                    )

                # Add local detection results to global results timeline (compensating for segment overlap)
                self.video_detection_results = np.append(self.video_detection_results, results[:, :math.ceil(results.shape[1] * 0.9)], axis=1)
                self.video_segment_index += 1

        # If recording timed segments, plot global video detection results over all clips in timeline
        if time_indexed_files:
            global_start_time = datetime.strptime(video_segment_paths[0].split('/')[-1].replace('.mp4', '').split('_')[1], '%H:%M:%S.%f')
            global_end_time = timestamps[-1]
            print(f"Full timeline: {global_start_time.strftime('%H:%M:%S.%f')} => {global_end_time.strftime('%H:%M:%S.%f')}")
            self.plot_local_vqa(
                self.video_detection_results,
                true_time_labels=truth,
                startpoint=global_start_time, endpoint=global_end_time,
                output_file="motion-timeline.png"
            )

    def get_local_paths(self, dir, audio_detection=True, video_detection=True, time_indexed_files=True):
        sort_by_index = lambda path: int(path.split('/')[-1].split('_')[0][3:])
        audio_filenames, video_filenames = [], []

        if audio_detection:
            audio_dir = os.path.join(dir, "audio/*.wav")
            audio_filenames = glob.glob(audio_dir)

            if time_indexed_files: audio_filenames = list(sorted(audio_filenames, key=sort_by_index))

        if video_detection:
            video_dir = os.path.join(dir, "video/*.mp4")
            video_filenames = glob.glob(video_dir)
            if time_indexed_files: video_filenames = list(sorted(video_filenames, key=sort_by_index))

        return audio_filenames, video_filenames

    def get_local_audio(self, filename):
        # Retrieve and decode wav file from local storage
        samplerate, audio_asset = wavfile.read(filename)

        if len(audio_asset.shape) < 2:
            audio_asset = np.expand_dims(audio_asset, axis=0)

        no_channels = audio_asset.shape[1]
        length = audio_asset.shape[0] / samplerate

        return audio_asset

    def get_local_video(self, filename):
        # Retrieve and decode mp4 file from local storage
        video_source = cv2.VideoCapture(filename)
        frame_buffer = []
        success = True

        while success:
            # Read video frame-by-frame from the opencv capture object; img is (H, W, C)
            success, frame = video_source.read()
            if success:
                frame_buffer.append(frame)

        video_asset = np.stack(frame_buffer, axis=0)  # dimensions (T, H, W, C)

        return video_asset

    def audio_detection(self, audio_content, time_indexed_audio=False, detect_gaps=True, detect_clicks=False, plot=False, start_time=0, end_time=0, output_dir='./'):
        time_indexed_audio = time_indexed_audio and start_time != 0 and end_time != 0
        if time_indexed_audio:
            audio = []
            for time, chunk in audio_content:
                audio = np.append(audio, chunk, axis=1)

            start_time = audio_content[0][0]
            end_time = audio_content[-1][0]
        else:
            audio = audio_content

        detected_audio_gaps, detected_audio_clicks = self.audio_detector.process(
            audio,
            start_time=start_time,
            gap_detection=detect_gaps,
            click_detection=detect_clicks
        )

        print(f"\n * Audio detection (segment {self.audio_segment_index}):")
        if time_indexed_audio: print(f"     * Segment time range         : {start_time.strftime('%H:%M:%S.%f')} => {end_time.strftime('%H:%M:%S.%f')}")
        if detect_gaps: print(f"     * Detected gap times         : {[(s.strftime('%H:%M:%S'), e.strftime('%H:%M:%S')) for s, e in detected_audio_gaps]}")
        if detect_clicks: print(f"     * Detected click times       : {detected_audio_clicks}")

        # Plot audio signal and any detections
        if plot:
            self.plot_audio(audio, detected_audio_gaps, detected_audio_clicks, start_time, end_time, time_indexed_audio, output_dir)

        print()
        return { "gaps": detected_audio_gaps, "clicks": detected_audio_clicks }

    def plot_audio(self, audio_content, gap_times, click_times, startpoint, endpoint, time_indexed_files, output_path):
        # Setup
        plt.rcParams['agg.path.chunksize'] = 1000
        fig, axs = plt.subplots(1, figsize=(20, 10), tight_layout=True)

        # Form timeline over clip
        time_x = np.linspace(0, 1, len(audio_content[0])) * (endpoint - startpoint) + startpoint
        time_index = np.linspace(0, len(audio_content[0]), len(audio_content[0]))

        # Plot L/R/Mono channels
        for idx, audio_channel in enumerate(audio_content):
            axs.plot(time_index, audio_channel, color='k', alpha=0.5, linewidth=0.5, label=f"Channel {idx}")

        # Plot time range of any audio gaps
        if len(gap_times) > 0:
            for start, end in gap_times:
                approx_gap_start = min(time_x, key=lambda dt: abs(dt - start))
                approx_gap_start_idx = np.where(time_x == approx_gap_start)[0][0]
                approx_gap_end = min(time_x, key=lambda dt: abs(dt - end))
                approx_gap_end_idx = np.where(time_x == approx_gap_end)[0][0]

                line = axs.axvspan(approx_gap_start_idx, approx_gap_end_idx, color='b', alpha=0.3)

            line.set_label('Detected gap')

        # Plot time range of any click artefacts
        if len(click_times) > 0:
            for time in click_times:
                approx_click_time = min(time_x, key=lambda dt: abs(dt - time))
                approx_click_idx = np.where(time_x == approx_click_time)[0][0]
                line = axs.axvline(approx_click_idx, color='r', linewidth=1)

            line.set_label('Detected click')

        axs.set_xticks(time_index[::self.audio_fps])
        if time_indexed_files:
            axs.set_xticklabels([t.strftime('%H:%M:%S') for t in time_x[::self.audio_fps]], fontsize=12)
        else:
            axs.set_xticklabels(time_x[::self.audio_fps], fontsize=12)

        plt.yticks(fontsize=12)

        plt.xlabel("\nCapture Time (H:M:S)", fontsize=14)
        plt.ylabel("Audio Sample Amplitude", fontsize=14)
        plt.legend(loc=1, fontsize=14)

        if time_indexed_files:
            plt.title(f"Audio Defect Detection: Segment {self.audio_segment_index} ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')})) \n", fontsize=18)
        else:
            plt.title(f"Audio Defect Detection \n", fontsize=18)

        # Save plot to file
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_path, f"audio-plot-{self.audio_segment_index}.png")

        print(f"     * Audio plot generated : {output_path}")
        fig.savefig(output_path)
        plt.close(fig)

    def video_detection(self, video_content, time_indexed_video=False, plot=False, start_time=0, end_time=0, epochs=1, output_dir='./'):
        time_indexed_video = time_indexed_video and start_time != 0 and end_time != 0
        if time_indexed_video:
            video = []
            for time, frame in video_content:
                video = np.append(video, frame, axis=1)

            start_time = video_content[0][0]
            end_time = video_content[-1][0]
        else:
            video = video_content

        # MaxVQA AI detection process
        print(f"\n * Video detection (segment {self.video_segment_index}):")

        processing_time_start = timer()
        scores = np.zeros(shape=(epochs,), dtype=object)
        for i in range(epochs):
            score_per_patch = self.video_detector.process(video_content)
            scores[i] = np.array(score_per_patch)

        processing_time_end = timer() - processing_time_start
        score_per_patch = np.mean(scores, axis=0)
        local_scores = np.mean(score_per_patch, axis=0)
        global_scores = np.mean(local_scores, axis=1)
        output = local_scores

        print(f"     * Global VQA scores  : {np.array([f'{i}: {s:.2f}' for i, s in enumerate(global_scores)], dtype=str)}")
        print(f"     * Processing time    : {processing_time_end:.2f}s")

        if plot:
            self.plot_local_vqa(local_scores, startpoint=start_time, endpoint=end_time, output_path=output_dir)

        print()
        return output

    def plot_local_vqa(self, vqa_values, true_time_labels=None, startpoint=0, endpoint=0, plot_motion_only=True, output_path='./'):
        # Metrics & figure setup
        if plot_motion_only:
            priority_metrics = [14]
            titles = {
                "A": "Motion fluency"
            }
            plot_values = vqa_values[priority_metrics]
            fig, axes = plt.subplot_mosaic("A", sharex=True, sharey=True, figsize=(12, 6), tight_layout=True)
        else:
            priority_metrics = [7, 9, 11, 13, 14]
            titles = {
                "A": "Sharpness",
                "B": "Noise",
                "C": "Flicker",
                "D": "Compression artefacts",
                "E": "Motion fluency"
            }
            fig, axes = plt.subplot_mosaic("AB;CD;EE", sharex=True, sharey=True, figsize=(12, 9), tight_layout=True)

        colours = cycle(mcolors.TABLEAU_COLORS)

        # Timestamps
        time_indexed_files = startpoint != 0 and endpoint != 0
        if time_indexed_files:
            time_x = np.linspace(0, 1, len(plot_values[0])) * (endpoint - startpoint) + startpoint
            time_index = np.linspace(0, len(plot_values[0]), len(plot_values[0]))

        for value_id, (ax_id, title) in enumerate(titles.items()):
            # Plot true values of known video defect times if they exist
            if time_indexed_files and true_time_labels is not None:
                for times in true_time_labels:
                    start = datetime.strptime(times[0], '%H:%M:%S')
                    approx_start = min(time_x, key=lambda dt: abs(dt - start))
                    approx_start_idx = np.where(time_x == approx_start)[0][0]

                    end = datetime.strptime(times[-1], '%H:%M:%S')
                    approx_end = min(time_x, key=lambda dt: abs(dt - end))
                    approx_end_idx = np.where(time_x == approx_end)[0][0]

                    axes[ax_id].axvspan(approx_start_idx, approx_end_idx, facecolor='grey', alpha=0.3, label="True stuttering")

            # Plot mean and twice standard deviation of VQA scores of each metric
            mean_over_video = plot_values[value_id].mean()
            std_over_video = plot_values[value_id].std()

            axes[ax_id].set_title(title)
            axes[ax_id].grid(linewidth=0.2)

            axes[ax_id].axhline(mean_over_video, color='black', ls='--', linewidth=0.5, label="Mean score")
            axes[ax_id].axhline(mean_over_video - 2 * std_over_video, color='black', ls='--', linewidth=0.5, label="Two standard deviations")

            # Plot VQA scores themselves
            if time_indexed_files:
                axes[ax_id].plot(time_index, plot_values[value_id], linewidth=0.75, color=next(colours), label="Score")
            else:
                axes[ax_id].plot(plot_values[value_id], linewidth=0.75, color=next(colours), label="Score")

        # Format title and axes labels
        if time_indexed_files:
            fig.suptitle(f"MaxVQA Video Defect Detection: Segment {self.video_segment_index}' ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')})", fontsize=16)
            fig.supxlabel("Capture Time (H:M:S)")
            num_ticks = round(len(plot_values[0])/10)
            plt.xticks(
                ticks=time_index[::num_ticks],
                labels=[t.strftime('%H:%M:%S') for t in time_x[::num_ticks]]
            )
        else:
            fig.suptitle("MaxVQA Video Defect Detection", fontsize=16)
            fig.supxlabel("Capture Frame")

        fig.supylabel("Absolute score (0-1, bad-good)")
        plt.yticks([0, 0.25, 0.5, 0.75, 1])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc=4)

        for ax in fig.get_axes():
            ax.label_outer()

        # Save plot to file
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_path, f"motion-plot-{self.video_segment_index}.png")

        print(f"     * Video plot generated : {output_path}")
        fig.savefig(output_path)
        plt.close(fig)


if __name__ == '__main__':
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='StutterDetection.py',
        description='Run audio and video stutter detection algorithms over local AV segments.'
    )

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(ROOT_DIR, "output/capture/")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output/stutter_detection/")

    parser.add_argument('-i', '--input', default=INPUT_DIR)
    parser.add_argument('-o', '--output', default=OUTPUT_DIR)
    parser.add_argument('-na', '--no-audio', action='store_false', default=True, help="Do not perform stutter detection on the audio track")
    parser.add_argument('-nv', '--no-video', action='store_false', default=True, help="Do not perform stutter detection on the video track")
    parser.add_argument('-c', '--clean-video', action='store_true', default=False, help="Testing on clean stutter-free videos (for experimentation)")
    parser.add_argument('-t', '--true-timestamps', action='store_true', default=False, help="Plot known stutter times on the output graph, specified in 'true-stutter-timestamps.json")
    parser.add_argument('-x', '--time-indexed-files', action='store_true', default=False, help="Label batch of detections over video segments with their time range (from filename)")
    parser.add_argument('-f', '--frames', type=int, default=256, help="Number of frames to downsample video to")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of times to repeat inference per video")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Specify processing hardware")

    # Decode input parameters to toggle between cameras, microphones, and setup mode.
    args = parser.parse_args()

    path = args.input
    out_path = args.output
    frames = args.frames
    epochs = args.epochs
    device = args.device
    stutter = not args.clean_video
    audio_on = args.no_audio
    video_on = args.no_video
    plot_true_timestamps = args.true_timestamps
    index_by_file_timestamp = args.time_indexed_files

    # Initialise and run Stutter Detection module
    detector = StutterDetection(video_downsample_frames=frames, device=device)

    if path.endswith(".mp4") or path.endswith(".wav"):
        detector.process(
            directory_path=path,
            time_indexed_files=index_by_file_timestamp,
            inference_epochs=epochs,
            audio_detection=audio_on,
            video_detection=video_on,
            output_directory=out_path
        )
    else:
        if plot_true_timestamps:
            timestamps_file = f"{path}/true-stutter-timestamps.json"
            if not os.path.isfile(timestamps_file):
                print(f"Error: no true timestamps file found but 'plot_true_timestamps' enabled. Checked location: {timestamps_file}")
                exit(1)

            with open(timestamps_file, 'r') as f:
                json_data = json.load(f)
                true_timestamps_json = json_data["timestamps"]

            detector.process(
                directory_path=path,
                truth=true_timestamps_json,
                time_indexed_files=index_by_file_timestamp,
                inference_epochs=epochs,
                audio_detection=audio_on,
                video_detection=video_on,
                output_directory=out_path
            )
        else:
            detector.process(
                directory_path=path,
                time_indexed_files=index_by_file_timestamp,
                inference_epochs=epochs,
                audio_detection=audio_on,
                video_detection=video_on,
                output_directory=out_path
            )
