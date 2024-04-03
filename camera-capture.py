import datetime
import math
import signal
from collections import deque
from threading import Thread
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pyaudio
import sounddevice


class VideoStream():
    def __init__(self, device=0):
        # Define a video capture object
        self.video_device = device
        self.video_stream = cv2.VideoCapture(self.video_device)
        self.frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.stream_open = False

    def launch(self, frame_queue, display_stream=False):
        self.stream_open = True
        # Show the video stream (no processing)
        if display_stream:
            while self.video_stream.isOpened():
                _, frame = self.video_stream.read()
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
                cv2.imshow('Video Stream', frame)
                # print(f"Timestamp counter: {timestamp}", end='\r')

                # esc to quit
                if cv2.waitKey(1) == 27:
                    self.kill()
                    break

        # Conduct video processing (save frames to queue instead of displaying)
        else:
            # Flush initial two black frames
            self.video_stream.read()
            self.video_stream.read()

            while self.stream_open:
                # Capture the video frame by frame
                _, frame = self.video_stream.read()
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
                frame_queue.append((timestamp, frame))
                print(f"Timestamp counter: {timestamp}", end='\r')

        self.video_stream.release()
        cv2.destroyAllWindows()
        print("\nVideo thread ended.")

    def kill(self):
        self.stream_open = False


class AudioStream():
    def __init__(self, device=1, sample_rate=44100):
        self.format = pyaudio.paInt16
        self.rate = sample_rate
        self.chunk = 1024
        self.stream = None
        self.stream_open = False

        self.audio_device = device
        self.audio = pyaudio.PyAudio()
        self.audio_channels = self.audio.get_device_info_by_host_api_device_index(0, self.audio_device).get('maxInputChannels')

        print(f'     * Audio input device         :', self.audio_device)
        print(f'     * Audio input channels       :', self.audio_channels)

    def launch(self, frame_queue):
        # Start audio recording
        self.stream_open = True
        stream = self.audio.open(
            format=self.format, rate=self.rate, input=True,
            input_device_index=self.audio_device, channels=self.audio_channels,
            frames_per_buffer=self.chunk
        )

        while self.stream_open:
            frame = stream.read(self.chunk)
            timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
            frame_queue.append((timestamp, frame))
            print(f"Timestamp counter: {timestamp}", end='\r')
            # print(f"Audio frame counter: {len(frame_queue) * self.chunk}", end='\r')

        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        print("Audio thread ended.")

    def kill(self):
        self.stream_open = False


class AudioVisualProcessor():
    def __init__(self, video_fps=30, audio_fps=44100,
                 audio_buffer_len_s=10, audio_overlap_len_s=2,
                 video_buffer_len_s=10, video_overlap_len_s=2):

        self.video_fps = video_fps
        self.audio_fps = audio_fps
        self.audio_segment_index = 0
        self.video_segment_index = 0

        self.audio_buffer_len_f = math.ceil(audio_fps * audio_buffer_len_s)
        self.audio_overlap_len_f = math.ceil(audio_fps * audio_overlap_len_s)
        self.video_buffer_len_f = math.ceil(video_fps * video_buffer_len_s)
        self.video_overlap_len_f = math.ceil(video_fps * video_overlap_len_s)

    def process(self, audio_module, video_module, audio_frames, video_frames):
        print(f"     * Audio segment size         : {self.audio_buffer_len_f}")
        print(f"     * Audio overlap size         : {self.audio_overlap_len_f}")
        print(f"     * Video capture source       : {video_module.video_device}")
        print(f"     * Video frame rate           : {self.video_fps}")
        print(f"     * Video segment size         : {self.video_buffer_len_f}")
        print(f"     * Video overlap size         : {self.video_overlap_len_f}", end='\n\n')

        while (audio_module.stream_open and video_module.stream_open) or \
            (len(audio_frames) >= self.audio_buffer_len_f) or \
            (len(video_frames) >= self.video_buffer_len_f):

            # Audio processing module
            if len(audio_frames) >= self.audio_buffer_len_f:
                audio_segment = self.collate_frames(audio_frames, av_type='audio')
                self.audio_detection(audio_segment)
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                video_segment = self.collate_frames(video_frames, av_type='video')
                self.video_detection(video_segment)
                self.video_segment_index += 1

        print(f"\n\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video")

    def collate_frames(self, frame_queue, av_type='audio'):
        if av_type == 'audio':
            core_size = self.audio_buffer_len_f - self.audio_overlap_len_f
            overlap_size = self.audio_overlap_len_f
            index = self.audio_segment_index
        else:
            core_size = self.video_buffer_len_f - self.video_overlap_len_f
            overlap_size = self.video_overlap_len_f
            index = self.video_segment_index

        print(f"\n\n * New {av_type} segment ({index}):")
        print(f"     * Input queue length  : {len(frame_queue)}")

        # Add main frames in video segment to buffer
        frame_buffer = []
        for _ in range(core_size):
            frame_buffer.append(frame_queue.popleft())

        # Add overlap frames to buffer
        frame_buffer.extend([frame_queue[i] for i in range(overlap_size)])

        print(f"     * Segment frame count : {len(frame_buffer)}")
        print(f"     * Segment time range  : {frame_buffer[0][0]} => {frame_buffer[-1][0]}")
        print(f"     * Output queue length : {len(frame_queue)}", end='\n\n')

        return frame_buffer

    def audio_detection(self, audio_content):
        # fig = plt.figure()
        # s = fig.add_subplot(111)
        amplitude = np.frombuffer(b''.join([x[1] for x in audio_content]), np.int16)
        # s.plot(amplitude)
        # fig.savefig('audio.png')

        print(f"\n\n * Audio detection ({self.audio_segment_index}):")
        print(f"     * Audio amplitudes    : {amplitude}")

    def video_detection(self, video_content):
        # Detect average brightness
        brightness = []
        black_frame_detected = False

        for time, frame in video_content:
            average_brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
            brightness.append(average_brightness)
            if average_brightness < 10: black_frame_detected = True

        print(f"\n\n * Video detection ({self.video_segment_index}):")
        print(f"     * Average brightness  : {np.average(brightness):.2f}")
        print(f"     * Black frame detected: {black_frame_detected}", end='')


def signal_handler(sig, frame):
    print("\nCtrl+C detected. Stopping all AV threads.")
    audio.kill()
    video.kill()


if __name__ == '__main__':
    # Switch between setup video stream & processing stream
    setup_mode_only = False
    audio_device = 1
    video_device = 0

    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("\nAudio devices available: \n", sounddevice.query_devices())
    print(f"\n * Parameters:")
    print(f"     * Setup mode (no processing) : {setup_mode_only}")

    # Set up audio and video streams
    audio = AudioStream(device=audio_device)
    video = VideoStream(device=video_device)

    if setup_mode_only:
        # Set up stream to show content but do no processing
        print()
        video.launch(display_stream=True)
    else:
        # Initialise video stream
        audio_frame_queue = deque()
        video_frame_queue = deque()

        # Set up and launch audio-video stream thread and processing thread
        audio_thread = Thread(target=audio.launch, args=(audio_frame_queue,))
        video_thread = Thread(target=video.launch, args=(video_frame_queue,))

        audio_thread.start()
        video_thread.start()

        processor = AudioVisualProcessor(video_fps=video.frame_rate)
        processor.process(audio, video, audio_frame_queue, video_frame_queue)
