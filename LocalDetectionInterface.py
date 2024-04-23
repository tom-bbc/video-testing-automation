import numpy as np
import wave
from datetime import datetime
import boto3
import cv2
import os
import json
import math
import glob
from scipy.io import wavfile

from AudioVisualDetector import AudioVisualDetector


class LocalDetectionInterface(AudioVisualDetector):
    def __init__(self, *args, **kwargs):
        super(LocalDetectionInterface, self).__init__(*args, **kwargs)
        self.audio_detection_results = []
        self.video_detection_results = np.array([[]]*16)

    def process(self, directory_path, truth=None, audio_detection=True, video_detection=True, plot=True, time_indexed_files=True, inference_epochs=1):
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
            audio_segment_paths, video_segment_paths = self.get_local_paths(audio_detection, video_detection, dir=directory_path)
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
                        end_time=timestamps[-1]
                    )
                else:
                    results = self.audio_detection(
                        audio_segment,
                        plot=plot
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
                        epochs=inference_epochs
                    )
                    # print(f" * Video detection results: {results.shape}")
                else:
                    results = self.video_detection(
                        video_segment,
                        plot=plot,
                        epochs=inference_epochs
                    )

                # Add local detection results to global results timeline (compensating for segment overlap)
                self.video_detection_results = np.append(self.video_detection_results, results[:, :math.ceil(results.shape[1] * 0.9)], axis=1)
                self.video_segment_index += 1

        # Plot global video detection results over all clips in timeline
        global_start_time = datetime.strptime(video_segment_paths[0].split('/')[-1].replace('.mp4', '').split('_')[1], '%H:%M:%S.%f')
        global_end_time = timestamps[-1]
        print(f"Full timeline: {global_start_time.strftime('%H:%M:%S.%f')} => {global_end_time.strftime('%H:%M:%S.%f')}")
        self.plot_local_vqa(
            self.video_detection_results,
            true_time_labels=truth,
            startpoint=global_start_time, endpoint=global_end_time,
            output_file="motion-timeline.png"
        )

    def get_local_paths(self, audio_detection=True, video_detection=True, dir="./output/data/"):
        sort_by_index = lambda path: int(path.split('/')[-1].split('_')[0][3:])
        audio_filenames, video_filenames = [], []

        if audio_detection:
            audio_filenames = glob.glob(f"{dir}*.wav")
            audio_filenames = list(sorted(audio_filenames, key=sort_by_index))

        if video_detection:
            video_filenames = glob.glob(f"{dir}*.mp4")
            video_filenames = list(sorted(video_filenames, key=sort_by_index))

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


if __name__ == '__main__':
    FRAMES = 256
    EPOCHS = 3
    PATH = "./output/data/uhd-nature"
    STUTTER = True

    detector = LocalDetectionInterface(video_downsample_frames=FRAMES, device='cpu')

    if STUTTER:
        with open(f"{PATH}/true-stutter-timestamps.json", 'r') as f:
            json_data = json.load(f)
            true_timestamps_json = json_data["timestamps"]

        detector.process(
            directory_path=f"{PATH}/stutter/",
            truth=true_timestamps_json,
            time_indexed_files=True,
            inference_epochs=EPOCHS
        )
    else:
        detector.process(
            directory_path=f"{PATH}/original/",
            time_indexed_files=True,
            inference_epochs=EPOCHS
        )
