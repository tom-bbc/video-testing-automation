import numpy as np
import wave
import time
import boto3

# from AudioVisualDetector import AudioVisualDetector


class Detector():
    def __init__(self, aws_access_key, aws_secret_key, *args, **kwargs):
        # super(Detector, self).__init__(*args, **kwargs)
        self.s3_bucket = 'av-segment-queue'
        self.aws_session = boto3.session.Session()
        self.s3_client = self.aws_session.client(
            service_name='s3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            endpoint_url="https://object.lon1.bbcis.uk"
        )

    def process(self):
        # Get list of AV files that currently are held in s3
        audio_segment_paths, video_segment_paths = self.get_av_filenames()

        # if len(audio_segment_paths) == 0 and len(video_segment_paths) == 0:
        #     time.sleep(5)

        # Cycle through each AV file running detection algorithms
        for index in range(max(len(audio_segment_paths), len(video_segment_paths))):
            # Run audio detection
            if index < len(audio_segment_paths):
                audio_path = audio_segment_paths[index]
                audio_segment = self.get_s3_file(audio_path)
                print("New audio segment:", audio_segment)
                # self.audio_detection(audio_segment, plot=False)
                # self.audio_segment_index += 1

            # Run video detection
            if index < len(video_segment_paths):
                video_path = video_segment_paths[index]
                video_segment = self.get_s3_file(video_path)
                print("New video segment:", video_segment)
                # self.video_detection(video_segment, plot=False)
                # self.video_segment_index += 1

    def get_av_filenames(self):
        # s3_buckets = s3_client.list_buckets()['Buckets']
        # for bucket in s3_buckets:
        #     print(f"Bucket: {bucket['Name']}")

        sort_by_modified_date = lambda obj: int(obj['LastModified'].strftime('%s'))

        # Get most recent audio segment files
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='audio-segments/')

        if response['KeyCount'] > 0:
            objects = response['Contents']
            audio_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]
            # for i, name in enumerate(audio_filenames):
            #     print(f"{i}. {name}")
        else:
            audio_filenames = []

        # Get most recent video segment files
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix='video-segments/')

        if response['KeyCount'] > 0:
            objects = response['Contents']
            video_filenames = [obj['Key'] for obj in sorted(objects, key=sort_by_modified_date)]
            # for i, name in enumerate(video_filenames):
            #     print(f"{i}. {name}")
        else:
            video_filenames = []

        return audio_filenames, video_filenames

    def get_s3_file(self, filename):
        # Retrieve and decode file from s3
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=filename)
        byte_data = obj['Body'].read()
        media_asset = np.frombuffer(byte_data, np.float32)
        # ERROR HERE: `np.frombuffer` does not work for decoding/reading mp4 files as it produces a 1D array.

        # Delete file from s3 after reading it
        self.s3_client.delete_object(Bucket=self.s3_bucket, Key=filename)

        # wav_file = wave.open(f'test-audio-file.wav', 'wb')
        # wav_file.setnchannels(1)
        # wav_file.setsampwidth(2)
        # wav_file.setframerate(44100)
        # wav_file.writeframes(b''.join(audio_data))
        # wav_file.close()

        return media_asset


if __name__ == '__main__':
    aws_access_key = "ea749b0383ee4fc2a367c0f859fc1b68"
    aws_secret_key = "38619fd506354a90ae58d2feaceb5824"

    detector = Detector(aws_access_key, aws_secret_key)
    detector.process()
