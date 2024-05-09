import math
import cv2
import numpy as np
import pyaudio
import wave
import boto3
from moviepy.editor import VideoFileClip, AudioFileClip


Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualProcessor():
    def __init__(self, video_fps=30, video_shape=(), audio_fps=44100, audio_chunk_size=1024,
                 audio_buffer_len_s=10, audio_overlap_len_s=1,
                 video_buffer_len_s=10, video_overlap_len_s=1):

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

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                checkpoint_files=False,
                audio_on=True, video_on=True, synchronize=True):

        if audio_on:
            print(f"         * Segment size           : {self.audio_buffer_len_f}")
            print(f"         * Overlap size           : {self.audio_overlap_len_f}")

        if video_on:
            print(f"     * Video:")
            print(f"         * Capture device         : {video_module.video_device}")
            print(f"         * Frame rate             : {self.video_fps}")
            print(f"         * Segment size           : {self.video_buffer_len_f}")
            print(f"         * Overlap size           : {self.video_overlap_len_f}", end='\n\n')

        print(f"Start of audio-visual processing", end='\n\n')

        while (audio_module.stream_open or video_module.stream_open) or \
            (len(audio_frames) >= 2 * self.audio_buffer_len_f) or \
            (len(video_frames) >= 2 * self.video_buffer_len_f):

            # Syncronise audio and video frame queues
            if synchronize:
                if audio_on and video_on and len(audio_frames) >= 2 * self.audio_buffer_len_f and len(video_frames) >= 2 * self.video_buffer_len_f:
                    # Syncronisation
                    audio_frames, video_frames = self.sync_av_frame_queues(audio_frames, video_frames)

                    # Audio processing module
                    audio_segment, audio_file = self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps, checkpoint_files)
                    self.audio_segment_index += 1

                    # Video processing module
                    video_segment, video_file = self.collate_video_frames(video_frames, checkpoint_files)
                    self.video_segment_index += 1

                    # Combine audio and video into single syncronised file
                    if checkpoint_files:
                        audio_clip = AudioFileClip(f"output/capture/audio/{audio_file}")
                        video_clip = VideoFileClip(f"output/capture/video/{video_file}")

                        if video_clip.end < audio_clip.end:
                            audio_clip = audio_clip.subclip(0, video_clip.end)

                        full_clip = video_clip.set_audio(audio_clip)
                        full_clip.write_videofile(f"output/capture/segments/{video_file.replace('vid', 'seg')}", verbose=False, logger=None)
            else:
                # Audio processing module
                if len(audio_frames) >= self.audio_buffer_len_f:
                    _ = self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps, checkpoint_files)
                    self.audio_segment_index += 1

                # Video processing module
                if len(video_frames) >= self.video_buffer_len_f:
                    _ = self.collate_video_frames(video_frames, checkpoint_files)
                    self.video_segment_index += 1

        print(f"\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video \n")

    def sync_av_frame_queues(self, audio_frame_queue, video_frame_queue):
        audio_start_time, _ = audio_frame_queue[0]
        video_start_time, _ = video_frame_queue[0]
        new_audio_start_idx = None
        new_video_start_idx = None

        # Cycle through audio and video frames to get closest timestamp to start segment from
        if audio_start_time < video_start_time:
            # Throw away audio frames until we reach the closest video frame
            current_audio_idx = 1

            while current_audio_idx < len(audio_frame_queue):
                audio_timestamp, _ = audio_frame_queue[current_audio_idx]

                if audio_timestamp == video_start_time:
                    # Found an exact match between audio and video start times
                    new_audio_start_idx = current_audio_idx
                    break
                elif audio_timestamp > video_start_time:
                    # If we have passed the video start time, find the closest audio frame time before or after
                    prev_audio_timestamp, _ = audio_frame_queue[current_audio_idx - 1]
                    if abs(video_start_time - prev_audio_timestamp) <= abs(video_start_time - audio_timestamp):
                        new_audio_start_idx = current_audio_idx - 1
                    else:
                        new_audio_start_idx = current_audio_idx
                    break

                current_audio_idx += 1

        elif audio_start_time > video_start_time:
            # Throw away video frames until we reach the closest audio frame
            current_video_idx = 1

            while current_video_idx < len(video_frame_queue):
                video_timestamp, _ = video_frame_queue[current_video_idx]

                if video_timestamp == audio_start_time:
                    # Found an exact match between audio and video start times
                    new_video_start_idx = current_video_idx
                    break
                elif audio_timestamp > video_start_time:
                    # If we have passed the audio start time, find the closest video frame time before or after
                    prev_video_timestamp, _ = video_frame_queue[current_video_idx - 1]
                    if abs(audio_start_time - prev_video_timestamp) <= abs(audio_start_time - video_timestamp):
                        new_video_start_idx = current_video_idx - 1
                    else:
                        new_video_start_idx = current_video_idx
                    break

                current_video_idx += 1

        # Skip forward in the audio or video frame queue to synchronise start timestamps
        if new_audio_start_idx is not None:
            for _ in range(new_audio_start_idx): audio_frame_queue.popleft()

        if new_video_start_idx is not None:
            for _ in range(new_video_start_idx): video_frame_queue.popleft()

        print(f"\nSynchronised AV frame segments:")
        print(f" * Audio start time: {audio_frame_queue[0][0].strftime('%H:%M:%S.%f')}")
        print(f" * Video start time: {video_frame_queue[0][0].strftime('%H:%M:%S.%f')}")

        return audio_frame_queue, video_frame_queue

    def collate_audio_frames(self, frame_queue, no_channels=1, sample_rate=44100, save_wav_file=False):
        file_name = ''
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
            wav_file = wave.open(f'output/capture/audio/{file_name}', 'wb')
            wav_file.setnchannels(no_channels)
            wav_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(frame_bytes_buffer))
            wav_file.close()
            print(f" * Audio end time: {frame_buffer[-1][0].strftime('%H:%M:%S.%f')}")

        return np.array(frame_buffer, dtype=object), file_name

    def collate_video_frames(self, frame_queue, save_mp4_file=False):
        # Setup memory buffer of frames and output video file
        file_name = ''
        frame_buffer = []
        if save_mp4_file:
            file_name = f"vid{self.video_segment_index}_{frame_queue[0][0].strftime('%H:%M:%S.%f')}_{frame_queue[self.video_buffer_len_f - 1][0].strftime('%H:%M:%S.%f')}.mp4"
            output_file = cv2.VideoWriter(
                f"output/capture/video/{file_name}",
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
            print(f" * Video end time: {frame_buffer[-1][0].strftime('%H:%M:%S.%f')}", end='\n\n')

        return np.array(frame_buffer, dtype=object), file_name
