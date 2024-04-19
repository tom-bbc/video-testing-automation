import cv2
import sys
import math
import random
import argparse
from datetime import timedelta


def run(video_path, output_path, num_stutters, max_stutter_length=1, min_stutter_length=0.2):
    video = cv2.VideoCapture(video_path)

    length_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stutter_frames = sorted(random.sample(range(length_frames), num_stutters))
    f_idx = 0

    while f_idx < length_frames:
        _, frame = video.read()
        output_writer.write(frame)

        if len(stutter_frames) > 0 and f_idx == stutter_frames[0]:
            stutter_frames.pop(0)
            stutter_length_seconds = round(random.uniform(min_stutter_length, max_stutter_length), 2)
            stutter_length_frames = round(fps * stutter_length_seconds)

            # print(f"Stutter frame           : {f_idx}")
            print(f"Stutter start timestamp : {timedelta(seconds=math.floor(f_idx / fps))}")
            print(f"Stutter end timestamp   : {timedelta(seconds=math.ceil((f_idx / fps) + stutter_length_seconds))}")
            print(f"Stutter length          : {stutter_length_seconds}s", end='\n\n')

            # Write same repeated frame to video for entirety of stutter
            for _ in range(stutter_length_frames):
                if f_idx < length_frames:
                    output_writer.write(frame)
                    _ = video.read()
                    f_idx += 1
        f_idx += 1


if __name__ == '__main__':
        # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='generate_video_stutter.py',
        description='Generate artificial gaps/stutter in a given video file.'
    )

    parser.add_argument("input_path")
    parser.add_argument('-s', '--stutters', type=int, default=3)
    parser.add_argument('-l,', '--stutter-length', type=float, default=2.0)
    parser.add_argument('-o', '--output-path', type=str, default='output-video.mp4')

    # Decode input parameters to toggle between cameras, microphones, and setup mode.
    args = parser.parse_args()
    video_path = args.input_path
    stutters = args.stutters
    output_path = args.output_path
    stutter_length = args.stutter_length

    print("\nInput             :", video_path)
    print("Num stutters      :", stutters)
    print("Output            :", output_path)
    print("Length of stutter :", stutter_length, end='\n\n')

    run(video_path, output_path, stutters, stutter_length)
