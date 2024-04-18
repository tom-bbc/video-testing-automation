import numpy as np
import wave
from datetime import datetime
import boto3
import cv2
import os

import glob
from scipy.io import wavfile

from AudioVisualDetector import AudioVisualDetector


class CloudDetector(AudioVisualDetector):
    def __init__(self, aws_access_key, aws_secret_key, *args, **kwargs):
        super(CloudDetector, self).__init__(*args, **kwargs)
        self.s3_bucket = 'video-testing-automation'
        self.aws_session = boto3.session.Session()
        self.s3_client = self.aws_session.client(
            service_name='s3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            endpoint_url="https://object.lon1.bbcis.uk"
        )

    def process(self, audio_detection=True, video_detection=True, location='local'):
        print(f"AV file locations: {location}")

        if location ==  's3':
            # Get list of AV files that currently are held in s3
            audio_segment_paths, video_segment_paths = self.get_s3_paths(audio_detection, video_detection)
        else:
            audio_segment_paths, video_segment_paths = self.get_local_paths(audio_detection, video_detection)

        # Cycle through each AV file running detection algorithms
        for index in range(max(len(audio_segment_paths), len(video_segment_paths))):
            # Run audio detection
            if audio_detection and index < len(audio_segment_paths):
                audio_path = audio_segment_paths[index]

                if location == 's3':
                    audio_segment = self.get_s3_audio(audio_path)
                else:
                    audio_segment = self.get_local_audio(audio_path)

                timestamps = [datetime.strptime(f, '%H:%M:%S.%f') for f in audio_path.split('/')[-1].replace('.wav', '').split('_')[1:]]
                print(f"New audio segment: {audio_path.split('/')[-1]} {audio_segment.shape}")

                results = self.audio_detection(
                    audio_segment,
                    plot=True,
                    start_time=timestamps[0],
                    end_time=timestamps[-1]
                )
                self.audio_segment_index += 1

            # Run video detection
            if video_detection and index < len(video_segment_paths):
                video_path = video_segment_paths[index]

                if location == 's3':
                    video_segment = self.get_s3_video(video_path)
                else:
                    video_segment = self.get_local_video(video_path)

                timestamps = [datetime.strptime(f, '%H:%M:%S.%f') for f in video_path.split('/')[-1].replace('.mp4', '').split('_')[1:]]
                print(f"New video segment: {video_path.split('/')[-1]} {video_segment.shape}")

                results = self.video_detection(
                    video_segment,
                    plot=True,
                    start_time=timestamps[0],
                    end_time=timestamps[-1]
                )
                self.video_segment_index += 1

    def get_s3_paths(self, audio_detection=True, video_detection=True):
        sort_by_modified_date = lambda obj: int(obj['LastModified'].strftime('%s'))
        audio_filenames, video_filenames = [], []

        # Get most recent audio segment files
        if audio_detection:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='audio-segments/')

            if response['KeyCount'] > 0:
                objects = response['Contents']
                audio_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]

        # Get most recent video segment files
        if video_detection:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='video-segments/')

            if response['KeyCount'] > 0:
                objects = response['Contents']
                video_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]

        return audio_filenames, video_filenames

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

    def get_s3_audio(self, filename, delete_remote=False):
        # Retrieve and decode wav file from s3
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=filename)
        byte_data = obj['Body'].read()
        audio_asset = np.frombuffer(byte_data, np.int16)
        audio_asset = np.expand_dims(audio_asset, axis=0)

        # Delete file from s3 after reading it
        if delete_remote:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=filename)

        return audio_asset

    def get_local_audio(self, filename):
        # Retrieve and decode wav file from local storage
        samplerate, audio_asset = wavfile.read(filename)

        if len(audio_asset.shape) < 2:
            audio_asset = np.expand_dims(audio_asset, axis=0)

        no_channels = audio_asset.shape[1]
        length = audio_asset.shape[0] / samplerate

        return audio_asset

    def get_s3_video(self, filename, delete_remote=False):
        # Retrieve and decode mp4 file from s3
        video_url = self.s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': self.s3_bucket, 'Key': filename}
        )

        video_source = cv2.VideoCapture(video_url)
        frame_buffer = []
        success = True

        while success:
            # Read video frame-by-frame from the opencv capture object; img is (H, W, C)
            success, frame = video_source.read()
            if success:
                frame_buffer.append(frame)

        video_asset = np.stack(frame_buffer, axis=0)  # dimensions (T, H, W, C)

        # Delete file from s3 after reading it
        if delete_remote:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=filename)

        return video_asset

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
    aws_access_key = "ea749b0383ee4fc2a367c0f859fc1b68"
    aws_secret_key = "38619fd506354a90ae58d2feaceb5824"

    detector = CloudDetector(aws_access_key, aws_secret_key, video_downsample_frames=256, device='cpu')
    detector.process()
