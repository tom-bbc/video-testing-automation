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

from essentia_audio_detection import AudioDetector
from maxvqa_video_detection import VideoDetector


Object = lambda **kwargs: type("Object", (), kwargs)

class VideoStream():
    def __init__(self, device=0):
        # Define a video capture object
        self.video_device = device
        self.video_stream = cv2.VideoCapture(self.video_device)
        self.frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.stream_open = False

    def launch(self, frame_queue=None, display_stream=False):
        self.stream_open = True
        # Show the video stream (no processing)
        if display_stream:
            while self.video_stream.isOpened():
                _, frame = self.video_stream.read()
                timestamp = datetime.datetime.now()
                cv2.imshow('Video Stream', frame)
                print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

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
                timestamp = datetime.datetime.now()
                frame_queue.append((timestamp, frame))
                # print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

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
            timestamp = datetime.datetime.now()

            frame_queue.append((timestamp, frame))
            # print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        print("Audio thread ended.")

    def kill(self):
        self.stream_open = False


class AudioVisualProcessor():
    def __init__(self, video_fps=30, audio_fps=44100, audio_chunk_size=1024,
                 audio_buffer_len_s=10, audio_overlap_len_s=2,
                 video_buffer_len_s=10, video_overlap_len_s=2):

        self.video_fps = video_fps
        self.audio_fps = audio_fps
        self.audio_segment_index = 0
        self.video_segment_index = 0

        self.chunk_size = audio_chunk_size
        self.audio_buffer_len_f = math.ceil(audio_fps * audio_buffer_len_s / audio_chunk_size)
        self.audio_overlap_len_f = math.ceil(audio_fps * audio_overlap_len_s / audio_chunk_size)
        self.video_buffer_len_f = math.ceil(video_fps * video_buffer_len_s)
        self.video_overlap_len_f = math.ceil(video_fps * video_overlap_len_s)

        self.detector = AudioDetector()

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[],
                video_module=Object(stream_open=False, video_device=None), video_frames=[]):
        print(f"     * Audio segment size         : {self.audio_buffer_len_f}")
        print(f"     * Audio overlap size         : {self.audio_overlap_len_f}")
        print(f"     * Video capture source       : {video_module.video_device}")
        print(f"     * Video frame rate           : {self.video_fps}")
        print(f"     * Video segment size         : {self.video_buffer_len_f}")
        print(f"     * Video overlap size         : {self.video_overlap_len_f}", end='\n\n')

        print(f"Start of audio-visual processing")

        while (audio_module.stream_open or video_module.stream_open) or \
            (len(audio_frames) >= self.audio_buffer_len_f) or \
            (len(video_frames) >= self.video_buffer_len_f):

            # Audio processing module
            # print(f"Audio frame counter: {len(audio_frames)}")
            if len(audio_frames) >= self.audio_buffer_len_f:
                audio_segment = self.collate_audio_frames(audio_frames)
                self.audio_detection(audio_segment)
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                video_segment = self.collate_video_frames(video_frames)
                self.video_detection(video_segment)
                self.video_segment_index += 1

        # Save all detection timestamps to CSV database
        if len(self.detector.gaps) > 0:
            detected_gap_timestamps = np.array([(s.strftime('%H:%M:%S.%f'), e.strftime('%H:%M:%S.%f')) for s, e in self.detector.gaps])
            np.savetxt("output/detected_gaps.csv", detected_gap_timestamps, delimiter=",", fmt='%s', header='Timestamp')

        if len(self.detector.clicks) > 0:
            detected_click_timestamps = np.array([t.strftime('%H:%M:%S.%f') for t in self.detector.clicks])
            np.savetxt("output/detected_clicks.csv", detected_click_timestamps, delimiter=",", fmt='%s', header='Timestamp')

        print(f"\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video \n")

    def collate_audio_frames(self, frame_queue):
        print(f"\n * New audio segment ({self.audio_segment_index}):")
        print(f"     * Input queue length  : {len(frame_queue)}")

        # Add main frames in video segment to buffer
        frame_buffer = []
        for _ in range(self.audio_buffer_len_f - self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue.popleft()
            frame = np.frombuffer(frame_bytes, np.int16).astype(np.float32)

            # Format stereo and mono channel audio into shape (1, 1024) or (2, 1024)
            if len(frame) == 2 * self.chunk_size:
                channel0 = frame[0::2]
                channel1 = frame[1::2]
                frame = np.array([channel0, channel1])
            else:
                frame = np.expand_dims(frame, axis=0)

            frame_buffer.append((timestamp, frame))

        # Add overlap frames to buffer
        for i in range(self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue[i]
            frame = np.frombuffer(frame_bytes, np.int16).astype(np.float32)

            if len(frame) == 2 * self.chunk_size:
                channel0 = frame[0::2]
                channel1 = frame[1::2]
                frame = np.array([channel0, channel1])
            else:
                frame = np.expand_dims(frame, axis=0)

            frame_buffer.append((timestamp, frame))

        print(f"     * Segment time range  : {frame_buffer[0][0].strftime('%H:%M:%S.%f')} => {frame_buffer[-1][0].strftime('%H:%M:%S.%f')}")
        print(f"     * Output queue length : {len(frame_queue)}", end='\n\n')

        return np.array(frame_buffer, dtype=object)

    def collate_video_frames(self, frame_queue):
        print(f"\n * New video segment ({self.video_segment_index}):")
        print(f"     * Input queue length  : {len(frame_queue)}")

        # Add main frames in video segment to buffer
        frame_buffer = []
        for _ in range(self.video_buffer_len_f - self.video_overlap_len_f):
            frame_buffer.append(frame_queue.popleft())

        # Add overlap frames to buffer
        frame_buffer.extend([frame_queue[i] for i in range(self.video_overlap_len_f)])

        print(f"     * Segment time range  : {frame_buffer[0][0].strftime('%H:%M:%S.%f')} => {frame_buffer[-1][0].strftime('%H:%M:%S.%f')}")
        print(f"     * Output queue length : {len(frame_queue)}", end='\n\n')

        return frame_buffer

    def audio_detection(self, audio_content, plot=True):
        no_channels = audio_content[0][1].shape[0]
        audio_y = np.array([[]] * no_channels)
        time_x = []

        for time, chunk in audio_content:
            time_x.extend([time] * chunk.shape[1])
            audio_y = np.append(audio_y, chunk, axis=1)

        detected_audio_gaps, detected_audio_clicks = self.detector.process(audio_y, start_time=audio_content[0][0])

        print(f" * Audio detection ({self.audio_segment_index}):")
        print(f"     * Average amplitude   : {np.average(np.abs(audio_y)):.2f}")
        print(f"     * Detected gap times  : {[(s.strftime('%H:%M:%S.%f')[:-4], e.strftime('%H:%M:%S.%f')[:-4]) for s, e in detected_audio_gaps]}")
        print(f"     * Detected click times: {[t.strftime('%H:%M:%S.%f')[:-4] for t in detected_audio_clicks]}")

        # Plot audio signal and any detections
        if plot:
            # Setup
            plt.rcParams['agg.path.chunksize'] = 1000
            fig, axs = plt.subplots(1, figsize=(20, 10), tight_layout=True)

            # Plot L/R/Mono channels
            for idx, audio_channel in enumerate(audio_y):
                time_index = np.linspace(0, len(time_x), len(time_x))
                axs.plot(time_index, audio_channel, color='k', alpha=0.5, linewidth=0.5, label=f"Channel {idx}")

            # axs.set_xticks(
            #     [time_x[0], time_x[round(len(time_x) / 5)], time_x[round(len(time_x) / 5) * 2], time_x[round(len(time_x) / 5) * 3], time_x[round(len(time_x) / 5) * 4], time_x[-1]],
            #     labels=[time_x[0], time_x[round(len(time_x) / 5)], time_x[round(len(time_x) / 5) * 2], time_x[round(len(time_x) / 5) * 3], time_x[round(len(time_x) / 5) * 4], time_x[-1]]
            # )

            # Plot timestamp of any audio clicks
            # if len(detected_audio_clicks) > 0:
            #     for time in detected_audio_clicks:
            #         line = axs.axvline(time.strftime('%H:%M:%S.%f')[:-4], color='r', linewidth=1)

            #     line.set_label('Detected click')

            # Plot time range of any audio gaps
            if len(detected_audio_gaps) > 0:
                for start, end in detected_audio_gaps:
                    approx_gap_start = min(time_x, key=lambda dt: abs(dt - start))
                    approx_gap_start_idx = time_x.index(approx_gap_start)
                    approx_gap_end = min(time_x, key=lambda dt: abs(dt - end))
                    approx_gap_end_idx = time_x.index(approx_gap_end)

                    line = axs.axvspan(approx_gap_start_idx, approx_gap_end_idx, color='b', alpha=0.3)

                line.set_label('Detected gap')

            plt.xlabel('Capture Time (H:M:S)')
            plt.ylabel('Audio Sample')
            plt.title(f"Audio Defect Detection: Segment {self.audio_segment_index} ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')}))")
            plt.legend(loc=1)
            fig.savefig(f"output/audio-plot-{self.audio_segment_index}.png")

            print(f"     * Plot generated      : 'audio-plot-{self.audio_segment_index}.png'")

    def video_detection(self, video_content, plot=True):
        # Detect average brightness
        brightness = []
        black_frame_detected = False

        # detector = VideoDetector()
        # print("Video detector initialised. Starting file processing.")
        # detector.process(video_content)

        for time, frame in video_content:
            average_brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
            brightness.append(average_brightness)
            if average_brightness < 10: black_frame_detected = True

        print(f" * Video detection ({self.video_segment_index}):")
        print(f"     * Average brightness  : {np.average(brightness):.2f}")
        print(f"     * Black frame detected: {black_frame_detected}")

        if plot:
            fig = plt.figure(figsize=(10, 7), tight_layout=True)
            axs = fig.add_subplot(111)

            axs.plot(brightness, linewidth=1)
            plt.xlabel('Capture Time (H:M:S)')
            plt.ylabel('Average Brightness')
            plt.xticks(
                ticks=range(0, len(video_content), 30),
                labels=[f[0].strftime('%H:%M:%S.%f')[:-4] for f in video_content[::30]],
                rotation=-90
            )
            plt.title(f"Video Defect Detection: Segment {self.video_segment_index} ({video_content[0][0].strftime('%H:%M:%S')} => {video_content[-1][0].strftime('%H:%M:%S')})")
            fig.savefig(f"output/video-plot-{self.video_segment_index}.png")

            print(f"     * Plot generated      : 'video-plot-{self.video_segment_index}.png'")


def signal_handler(sig, frame):
    print("\nCtrl+C detected. Stopping all AV threads.")
    if audio_on: audio.kill()
    if video_on: video.kill()


if __name__ == '__main__':
    # Switch between setup video stream & processing stream
    audio_on = True
    video_on = True
    setup_mode_only = False

    audio_device = 3
    video_device = 1

    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print(f"\nInitialising capture and processing modules", end='\n\n')
    print(f"Audio devices available: \n{sounddevice.query_devices()}", end='\n\n')
    print(f" * Parameters:")
    print(f"     * Setup mode (no processing) : {setup_mode_only}")

    # Set up audio and video streams
    if video_on: video = VideoStream(device=video_device)
    if audio_on and not setup_mode_only: audio = AudioStream(device=audio_device)

    if setup_mode_only and video_on:
        # Set up stream to show content but do no processing
        print()
        video.launch(display_stream=True)
    else:
        # Set up and launch audio-video stream threads
        if audio_on:
            audio_frame_queue = deque()
            audio_thread = Thread(target=audio.launch, args=(audio_frame_queue,))
            audio_thread.start()

        if video_on:
            video_frame_queue = deque()
            video_thread = Thread(target=video.launch, args=(video_frame_queue,))
            video_thread.start()

        # Initialise and launch AV processing module
        if audio_on and video_on:
            processor = AudioVisualProcessor(video_fps=video.frame_rate)
            processor.process(audio, audio_frame_queue, video, video_frame_queue)
        elif video_on:
            processor = AudioVisualProcessor(video_fps=video.frame_rate)
            processor.process(video_module=video, video_frames=video_frame_queue)
        elif audio_on:
            processor = AudioVisualProcessor()
            processor.process(audio_module=audio, audio_frames=audio_frame_queue)
        else:
            exit(0)
