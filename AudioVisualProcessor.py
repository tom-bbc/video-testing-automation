import math
import cv2
import numpy as np
import pyaudio
import wave
import boto3


Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualProcessor():
    def __init__(self, video_fps=30, video_shape=(), audio_fps=44100, audio_chunk_size=1024,
                 audio_buffer_len_s=10, audio_overlap_len_s=1,
                 video_buffer_len_s=10, video_overlap_len_s=1,
                 aws_access_key='', aws_secret_key=''):

        self.video_fps = video_fps
        self.audio_fps = audio_fps
        self.video_shape = video_shape
        self.audio_segment_index = 0
        self.video_segment_index = 0

        self.chunk_size = audio_chunk_size
        self.audio_buffer_len_f = math.ceil(audio_fps * audio_buffer_len_s / audio_chunk_size)
        self.audio_overlap_len_f = math.ceil(audio_fps * audio_overlap_len_s / audio_chunk_size)
        self.video_buffer_len_f = math.ceil(video_fps * video_buffer_len_s)
        self.video_overlap_len_f = math.ceil(video_fps * video_overlap_len_s)

        if aws_access_key != '' and aws_secret_key != '':
            self.s3_bucket = 'video-testing-automation'
            self.aws_session = boto3.session.Session()
            self.s3_client = self.aws_session.client(
                service_name='s3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                endpoint_url="https://object.lon1.bbcis.uk"
            )

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                checkpoint_files=False, checkpoint_to_s3=False):

        if audio_module.stream_open:
            print(f"         * Segment size           : {self.audio_buffer_len_f}")
            print(f"         * Overlap size           : {self.audio_overlap_len_f}")

        if video_module.stream_open:
            print(f"     * Video:")
            print(f"         * Capture device         : {video_module.video_device}")
            print(f"         * Frame rate             : {self.video_fps}")
            print(f"         * Segment size           : {self.video_buffer_len_f}")
            print(f"         * Overlap size           : {self.video_overlap_len_f}", end='\n\n')

        print(f"Start of audio-visual processing", end='\n\n')

        while (audio_module.stream_open or video_module.stream_open) or \
            (len(audio_frames) >= self.audio_buffer_len_f) or \
            (len(video_frames) >= self.video_buffer_len_f):

            # Audio processing module
            if len(audio_frames) >= self.audio_buffer_len_f:
                self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps, checkpoint_files, checkpoint_to_s3)
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                self.collate_video_frames(video_frames, checkpoint_files, checkpoint_to_s3)
                self.video_segment_index += 1

        print(f"\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video \n")

    def collate_audio_frames(self, frame_queue, no_channels=1, sample_rate=44100, save_wav_file=False, save_to_s3=False):
        frame_bytes_buffer = []
        timestamps = []

        # Add main frames in video segment to buffer
        for _ in range(self.audio_buffer_len_f - self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue.popleft()
            timestamps.append(timestamp)
            frame_bytes_buffer.append(frame_bytes)

         # Add overlap frames to buffer
        for i in range(self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue[i]
            timestamps.append(timestamp)
            frame_bytes_buffer.append(frame_bytes)

        # Decode all frames from byte representation and arrange into timestamped segment
        frame_buffer = []
        for timestamp, frame_bytes in zip(timestamps, frame_bytes_buffer):
            frame = np.frombuffer(frame_bytes, np.int16)

            # Format stereo and mono channel audio into shape (1, 1024) or (2, 1024)
            if len(frame) == 2 * self.chunk_size:
                channel0 = frame[0::2]
                channel1 = frame[1::2]
                frame = np.array([channel0, channel1])
            else:
                frame = np.expand_dims(frame, axis=0)

            frame_buffer.append((timestamp, frame))

        # Save audio data to WAV file for checking later
        if save_wav_file:
            file_name = f"aud{self.audio_segment_index}_{frame_buffer[0][0].strftime('%H:%M:%S.%f')}_{frame_buffer[-1][0].strftime('%H:%M:%S.%f')}.wav"
            wav_file = wave.open(f'output/data/{file_name}', 'wb')
            wav_file.setnchannels(no_channels)
            wav_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(frame_bytes_buffer))
            wav_file.close()

            if save_to_s3:
                print("Uploading audio to S3")
                self.s3_client.upload_file(
                    f"output/data/{file_name}",
                    self.s3_bucket,
                    f"audio-segments/{file_name}"
                )

        return np.array(frame_buffer, dtype=object)

    def collate_video_frames(self, frame_queue, save_mp4_file=False, save_to_s3=False):
        # Setup memory buffer of frames and output video file
        frame_buffer = []
        if save_mp4_file:
            file_name = f"vid{self.video_segment_index}_{frame_queue[0][0].strftime('%H:%M:%S.%f')}_{frame_queue[self.video_buffer_len_f - 1][0].strftime('%H:%M:%S.%f')}.mp4"
            output_file = cv2.VideoWriter(
                f"output/data/{file_name}",
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.video_fps,
                self.video_shape
            )

        # Add main frames in video segment to buffer
        for _ in range(self.video_buffer_len_f - self.video_overlap_len_f):
            frame = frame_queue.popleft()
            frame_buffer.append(frame)
            if save_mp4_file: output_file.write(frame[1])

        # Add overlap frames to buffer
        for i in range(self.video_overlap_len_f):
            frame = frame_queue[i]
            frame_buffer.append(frame)
            if save_mp4_file: output_file.write(frame[1])

        if save_mp4_file:
            output_file.release()

            if save_to_s3:
                self.s3_client.upload_file(
                    f"output/data/{file_name}",
                    self.s3_bucket,
                    f"video-segments/{file_name}"
                )

        return frame_buffer
