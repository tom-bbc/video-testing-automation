import math
import cv2
import numpy as np
import pyaudio
import wave
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualProcessor():
    def __init__(self, video_fps=30, video_shape=(), audio_fps=44100, audio_chunk_size=1024,
                 audio_buffer_len_s=20, audio_overlap_len_s=1,
                 video_buffer_len_s=20, video_overlap_len_s=1,
                 audio_save_path='', video_save_path=''):

        self.audio_fps = audio_fps
        self.video_fps = video_fps
        self.video_shape = video_shape
        self.audio_segment_index = 0
        self.video_segment_index = 0

        self.chunk_size = audio_chunk_size
        self.audio_buffer_len_f = math.ceil(audio_fps * audio_buffer_len_s / audio_chunk_size)
        self.audio_overlap_len_f = math.ceil(audio_fps * audio_overlap_len_s / audio_chunk_size)
        self.video_buffer_len_f = math.ceil(video_fps * video_buffer_len_s)
        self.video_overlap_len_f = math.ceil(video_fps * video_overlap_len_s)

        self.save_audio_files = True
        self.save_video_files = True
        self.audio_save_path = audio_save_path
        self.video_save_path = video_save_path

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                checkpoint_files=True, audio_on=True, video_on=True):

        self.save_audio_files = self.save_video_files = checkpoint_files

        if audio_on:
            print(f"         * Segment size           : {self.audio_buffer_len_f}")
            print(f"         * Overlap size           : {self.audio_overlap_len_f}")

        if video_on:
            print(f"     * Video:")
            print(f"         * Capture device         : {video_module.video_device}")
            print(f"         * Frame rate             : {self.video_fps}")
            print(f"         * Segment size           : {self.video_buffer_len_f}")
            print(f"         * Overlap size           : {self.video_overlap_len_f}", end='\n\n')

        print(f"\nStart of audio-visual processing", end='\n\n')

        while (audio_module.stream_open or video_module.stream_open) or \
            (len(audio_frames) > self.audio_buffer_len_f) or \
            (len(video_frames) > self.video_buffer_len_f):

            # Audio processing module
            if len(audio_frames) >= self.audio_buffer_len_f:
                self.collate_audio_frames(audio_frames, audio_channels)
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                self.collate_video_frames(video_frames)
                self.video_segment_index += 1

        print(f"\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video \n")

    def collate_audio_frames(self, frame_queue, no_channels=1):
        file_name = ''
        frame_bytes_buffer = []
        frame_buffer = np.array([[]] * no_channels)
        start_timestamp = frame_queue[0][0].strftime('%H:%M:%S.%f')
        end_timestamp = frame_queue[self.audio_buffer_len_f - 1][0].strftime('%H:%M:%S.%f')

        # Add main frames in video segment to buffer
        for _ in range(self.audio_buffer_len_f - self.audio_overlap_len_f):
            frame_bytes = frame_queue.popleft()[1]
            frame_bytes_buffer.append(frame_bytes)

        # Add overlap frames to buffer
        for i in range(self.audio_overlap_len_f):
            frame_bytes = frame_queue[i][1]
            frame_bytes_buffer.append(frame_bytes)

        # Save audio data to WAV file for checking later
        if self.save_audio_files:
            file_name = f"aud{self.audio_segment_index}_{start_timestamp}_{end_timestamp}.wav"
            wav_file = wave.open(f'{self.audio_save_path}{file_name}', 'wb')
            wav_file.setnchannels(no_channels)
            wav_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.audio_fps)
            wav_file.writeframes(b''.join(frame_bytes_buffer))
            wav_file.close()
        else:
            # Decode all frames from byte representation and arrange into timestamped segment
            for frame_bytes in frame_bytes_buffer:
                frame = np.frombuffer(frame_bytes, np.int16)

                # Format stereo and mono channel audio into shape (1, 1024) or (2, 1024)
                if len(frame) == 2 * self.chunk_size:
                    channel0 = frame[0::2]
                    channel1 = frame[1::2]
                    frame = np.array([channel0, channel1])
                    frame_buffer = np.append(frame_buffer, frame, axis=1)
                else:
                    frame = np.expand_dims(frame, axis=0)
                    frame_buffer = np.append(frame_buffer, frame, axis=1)

        print(f" * Audio end time: {end_timestamp}")

        return {
            'buffer': frame_buffer,
            'file': file_name
        }

    def collate_video_frames(self, frame_queue):
        # Setup memory buffer of frames and output video file
        file_name = ''
        frame_buffer = []
        start_timestamp = frame_queue[0][0].strftime('%H:%M:%S.%f')
        end_timestamp = frame_queue[self.video_buffer_len_f - 1][0].strftime('%H:%M:%S.%f')

        if self.save_video_files:
            file_name = f"vid{self.video_segment_index}_{start_timestamp}_{end_timestamp}.mp4"
            output_file = cv2.VideoWriter(
                f"{self.video_save_path}{file_name}",
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.video_fps,
                self.video_shape
            )

        # Add main frames in video segment to buffer
        for _ in range(self.video_buffer_len_f - self.video_overlap_len_f):
            frame = frame_queue.popleft()[1]
            if self.save_video_files:
                output_file.write(frame)
            else:
                frame_buffer.append(frame)

        # Add overlap frames to buffer
        for i in range(self.video_overlap_len_f):
            frame = frame_queue[i][1]
            if self.save_video_files:
                output_file.write(frame)
            else:
                frame_buffer.append(frame)

        if self.save_video_files: output_file.release()
        print(f" * Video end time: {end_timestamp}", end='\n\n')

        return {
            'buffer': frame_buffer,
            'file': file_name
        }
