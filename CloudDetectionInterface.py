import numpy as np
from datetime import datetime
import boto3
import math
import cv2
from AudioVisualDetector import AudioVisualDetector


class CloudDetectionInterface(AudioVisualDetector):
    def __init__(self, aws_access_key, aws_secret_key, *args, **kwargs):
        super(CloudDetectionInterface, self).__init__(*args, **kwargs)
        self.s3_bucket = 'video-testing-automation'
        self.aws_session = boto3.session.Session()
        self.s3_client = self.aws_session.client(
            service_name='s3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            endpoint_url="https://object.lon1.bbcis.uk"
        )
        self.audio_detection_results = []
        self.video_detection_results = np.array([[]]*16)

    def process(self, truth=None, audio_detection=True, video_detection=True, plot=True, time_indexed_files=True, inference_epochs=1):
        # Get list of AV files that currently are held in s3
        audio_segment_paths, video_segment_paths = self.get_s3_paths(audio_detection, video_detection)

        # Cycle through each AV file running detection algorithms
        for index in range(max(len(audio_segment_paths), len(video_segment_paths))):
            # Run audio detection
            if audio_detection and index < len(audio_segment_paths):
                audio_path = audio_segment_paths[index]
                audio_segment = self.get_s3_audio(audio_path)
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
                video_segment = self.get_s3_video(video_path)
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
            true_timestamps=truth,
            startpoint=global_start_time, endpoint=global_end_time,
            output_file="video-timeline.png"
        )

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


if __name__ == '__main__':
    aws_access_key = "ea749b0383ee4fc2a367c0f859fc1b68"
    aws_secret_key = "38619fd506354a90ae58d2feaceb5824"

    detector = CloudDetectionInterface(
        aws_access_key, aws_secret_key,
        video_downsample_frames=256, device='cpu'
    )
    detector.process(
        time_indexed_files=True,
        inference_epochs=3
    )
