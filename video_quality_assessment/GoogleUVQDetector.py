
import numpy as np
from time import time
from decord import VideoReader
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import gfile

from uvq import uvq_utils as utils


# Distortion label names
DISTORTION_TYPES = [
  "Other distortions",
  "Gaussian blur", "Lens blur", "Motion blur",
  "Color diffusion", "Color shift", "Color quantization", "Color saturation 1", "Color saturation 2",
  "JPEG2000 compression", "JPEG compression",
  "White noise", "White noise in color component", "Impulse noise", "Multiplicative noise", "Denoise",
  "Brighten", "Darken", "Mean shift",
  "Jitter", "Non-eccentricity patch", "Pixelate", "Quantization", "Color block",
  "High sharpen", "Contrast change"
]


# Define actual detection module to be used
class VideoDetector():
    def __init__(self):
        self.model_dir = "uvq/models"
        self.feature_dir = "features"
        self.output_dir = ''

    def process(self, video_path, plot=True):
        # Get video length
        vreader = VideoReader(video_path)
        video_length_frames = len(vreader)
        fps = vreader.get_avg_fps()
        video_length = video_length_frames // fps
        video_length = int(video_length)

        # Generate video id and output path
        video_id = video_path.split('/')[-1][:-4]

        self.output_dir = f"output/uvq-results/{video_id}"
        self.feature_dir = f"{self.output_dir}/features"

        if not gfile.IsDirectory(self.feature_dir):
            gfile.MakeDirs(self.feature_dir)

        # Extract features of the input video
        start = time()
        utils.generate_features(video_id, video_length, video_path, self.model_dir, self.feature_dir)
        time_features = time() - start

        # Predict VQA of video using model
        start = time()
        utils.prediction(video_id, video_length, self.model_dir, self.feature_dir, self.output_dir)
        time_prediction = time() - start

        # Output timing info and plot results
        print(f"\n\nTime (feature generation): {time_features:.2f}s")
        print(f"Time (prediction): {time_prediction:.2f}s")
        print(f"Time (total): {time_features + time_prediction:.2f}s")

        if plot:
            plot_path = f"{self.output_dir}/{video_id}_plot.png"
            features_dataset_path = f"{self.feature_dir}/{video_id}_label_distortion.csv"
            label_distortion = np.genfromtxt(features_dataset_path, delimiter=',')
            label_distortion = label_distortion.reshape((label_distortion.shape[0], 4, 26))

            self.plot(label_distortion, video_id, plot_path)

    def plot(self, label_distortion, video_name, plot_name='vqa-plot.png'):
        plt.rcParams.update({'font.size': 20, 'figure.figsize': (30, 25)})

        if label_distortion.shape != (label_distortion.shape[0], 4, 26):
            label_distortion = label_distortion.reshape((label_distortion.shape[0], 4, 26))

        fig, axes = plt.subplots(7, 4, sharex=True, sharey=True, tight_layout=True)
        axes = axes.reshape(-1)

        for metric in range(26):
            for block in range(4):
                axes[metric].set_title(DISTORTION_TYPES[metric])
                axes[metric].grid(linewidth=0.2)
                axes[metric].plot(range(label_distortion.shape[0]), label_distortion[:, block, metric], linewidth=1, alpha=0.5, label=f'Block {block}')

            axes[metric].plot(range(label_distortion.shape[0]), label_distortion[:, :, metric].mean(-1), linewidth=1, linestyle='dashed', color='k', label=f'Mean')

        axes[-2].remove()
        axes[-1].remove()

        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        fig.suptitle(f'UVQ Distortion of Video Artefacts ({video_name})\n', fontsize=45)
        fig.supylabel('Level of distortion (0-1, good-bad)\n',  fontsize=30)
        fig.supxlabel('Video time (s)',  fontsize=30)

        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1.05))
        plt.savefig(plot_name)
