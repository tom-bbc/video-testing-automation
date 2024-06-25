import math
import cv2
import numpy as np
import pyaudio
import wave
from moviepy.editor import VideoFileClip, AudioFileClip


Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualProcessor():
    def __init__(self, video_fps=30, video_shape=(), audio_fps=44100, audio_chunk_size=1024,
                 audio_buffer_len_s=10, audio_overlap_len_s=1,
                 video_buffer_len_s=10, video_overlap_len_s=1,
                 audio_save_path="output/capture/audio/",
                 video_save_path="output/capture/video/",
                 av_save_path="output/capture/segments/"
                ):

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

        self.audio_save_path = audio_save_path
        self.video_save_path = video_save_path
        self.segment_save_path = av_save_path

        self.save_wav_file = False
        self.save_mp4_file = False

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                checkpoint_files=False, audio_on=True, video_on=True, synchronize=True):

        self.save_wav_file = checkpoint_files
        self.save_mp4_file = checkpoint_files

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
                    collated_audio = self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps)
                    audio_file = collated_audio['file']
                    self.audio_segment_index += 1

                    # Video processing module
                    collated_video = self.collate_video_frames(video_frames)
                    video_file = collated_video['file']
                    self.video_segment_index += 1

                    # Combine audio and video into single syncronised file
                    if checkpoint_files:
                        audio_clip = AudioFileClip(f"{self.audio_save_path}{audio_file}")
                        video_clip = VideoFileClip(f"{self.video_save_path}{video_file}")

                        if video_clip.end < audio_clip.end:
                            audio_clip = audio_clip.subclip(0, video_clip.end)

                        full_clip = video_clip.set_audio(audio_clip)
                        full_clip.write_videofile(f"{self.segment_save_path}{video_file.replace('vid', 'seg')}", verbose=False, logger=None)
            else:
                # Audio processing module
                if len(audio_frames) >= self.audio_buffer_len_f:
                    self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps)
                    self.audio_segment_index += 1

                # Video processing module
                if len(video_frames) >= self.video_buffer_len_f:
                    self.collate_video_frames(video_frames)
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

        print(f"\n\nSynchronised AV frame segments:")
        print(f" * Audio start time: {audio_frame_queue[0][0].strftime('%H:%M:%S.%f')}")
        print(f" * Video start time: {video_frame_queue[0][0].strftime('%H:%M:%S.%f')}")

        return audio_frame_queue, video_frame_queue

    def collate_audio_frames(self, frame_queue, no_channels=1, sample_rate=44100):
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
        if self.save_wav_file:
            file_name = f"aud{self.audio_segment_index}_{start_timestamp}_{end_timestamp}.wav"
            wav_file = wave.open(f'{self.audio_save_path}{file_name}', 'wb')
            wav_file.setnchannels(no_channels)
            wav_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
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

        if self.save_mp4_file:
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
            if self.save_mp4_file:
                output_file.write(frame)
            else:
                frame_buffer.append(frame)

        # Add overlap frames to buffer
        for i in range(self.video_overlap_len_f):
            frame = frame_queue[i][1]
            if self.save_mp4_file:
                output_file.write(frame)
            else:
                frame_buffer.append(frame)

        if self.save_mp4_file: output_file.release()
        print(f" * Video end time: {end_timestamp}", end='\n\n')

        return {
            'buffer': frame_buffer,
            'file': file_name
        }
