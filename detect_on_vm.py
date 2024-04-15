import numpy as np
import wave
import time
import boto3
import cv2
import os

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

    def process(self, audio_detection=True, video_detection=True):
        # Get list of AV files that currently are held in s3
        paths = self.get_av_filenames(audio_detection, video_detection)
        audio_segment_paths, video_segment_paths = paths["audio"], paths["video"]

        # if len(audio_segment_paths) == 0 and len(video_segment_paths) == 0:
        #     time.sleep(5)

        # Cycle through each AV file running detection algorithms
        for index in range(max(len(audio_segment_paths), len(video_segment_paths))):
            # Run audio detection
            if audio_detection and index < len(audio_segment_paths):
                audio_path = audio_segment_paths[index]
                audio_segment = self.get_s3_audio(audio_path)
                print(f"New audio segment: {audio_path} {audio_segment.shape}")

                results = self.audio_detection(audio_segment, time_indexed_audio=False, plot=False)
                self.audio_segment_index += 1

            # Run video detection
            if video_detection and index < len(video_segment_paths):
                video_path = video_segment_paths[index]
                video_segment = self.get_s3_video(video_path)
                print(f"New video segment: {video_path} {video_segment.shape}")

                results = self.video_detection(video_segment, time_indexed_video=False, plot=False)
                self.video_segment_index += 1

    def get_av_filenames(self, audio_detection=True, video_detection=True):
        sort_by_modified_date = lambda obj: int(obj['LastModified'].strftime('%s'))
        filenames = {"audio": [], "video": []}

        # Get most recent audio segment files
        if audio_detection:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='audio-segments/')

            if response['KeyCount'] > 0:
                objects = response['Contents']
                audio_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]
                # for i, name in enumerate(audio_filenames):
                #     print(f"{i}. {name}")
                filenames["audio"] = audio_filenames

        # Get most recent video segment files
        if video_detection:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='video-segments/')

            if response['KeyCount'] > 0:
                objects = response['Contents']
                video_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]
                # for i, name in enumerate(video_filenames):
                #     print(f"{i}. {name}")
                filenames["video"] = video_filenames

        return filenames

    def get_s3_audio(self, filename, delete_remote=False, save_locally=False):
        # Retrieve and decode wav file from s3
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=filename)
        byte_data = obj['Body'].read()
        audio_asset = np.frombuffer(byte_data, np.int16)
        audio_asset = np.expand_dims(audio_asset, axis=0)

        # Delete file from s3 after reading it
        if delete_remote:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=filename)

        # TEST: save wav file
        if save_locally:
            wav_file = wave.open(os.path.basename(filename), 'wb')
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(b''.join(audio_asset))
            wav_file.close()

        return audio_asset

    def get_s3_video(self, filename, delete_remote=False, save_locally=False):
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

        # TEST: save mp4 file
        if save_locally:
            output_file = cv2.VideoWriter(
                f"test-video-file.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                video_source.get(cv2.CAP_PROP_FPS),
                (int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            for f in video_asset:
                output_file.write(f)

            output_file.release()

        return video_asset


if __name__ == '__main__':
    aws_access_key = "ea749b0383ee4fc2a367c0f859fc1b68"
    aws_secret_key = "38619fd506354a90ae58d2feaceb5824"

    detector = CloudDetector(aws_access_key, aws_secret_key)
    detector.process()
