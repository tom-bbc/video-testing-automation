import os
import glob
import argparse
import numpy as np
from GoogleUVQDetector import VideoDetector


class VideoQualityDetection():
    def __init__(self):
        self.video_detector = VideoDetector()
        self.video_detection_results = np.array([[]]*16)

    def process(self, directory_path, plot=True):
        if os.path.isfile(directory_path) and directory_path.endswith(".mp4"):
            # Permits running on single input file
            video_segment_paths = [directory_path]
        elif os.path.isdir(directory_path):
            # Gets list of AV files from local directory
            video_segment_paths = self.get_local_paths(directory_path)
        else:
            print(f"No input video found at path: {directory_path}")
            exit(1)

        # Cycle through each AV file running detection algorithms
        for video_path in video_segment_paths:
            # Run video detection
            print(f"New video segment: {video_path.split('/')[-1]}")
            self.video_detector.process(video_path, plot=plot)

    def get_local_paths(self, dir="./data/"):
        sort_by_index = lambda path: int(path.split('/')[-1].split('_')[0][3:])
        video_filenames = [], []
        video_filenames = glob.glob(f"{dir}*.mp4")
        video_filenames = list(sorted(video_filenames, key=sort_by_index))

        return video_filenames


if __name__ == "__main__":
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='VideoQualityDetection.py',
        description='Run video video quality assessment using Google UVQ over local videos.'
    )

    parser.add_argument("path")
    args = parser.parse_args()
    video = args.path

    detector = VideoQualityDetection()
    detector.process(video)
