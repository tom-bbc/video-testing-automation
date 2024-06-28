import cv2
import ffmpeg
import pyaudio
import datetime


class CombinedCaptureStream():
    def __init__(self, audio_source=0, video_source=0, checkpoint_path='',
                 segment_length_s=10):

        # Check device indices with cmd: `ffmpeg -hide_banner -list_devices true -f avfoundation -i ''`
        self.audio_device = audio_source
        self.video_device = video_source

        self.video_fps = 30
        self.video_width = 1280
        self.video_height = 720

        self.save_path = checkpoint_path
        self.segment_length_s = segment_length_s

    def launch(self, total_processing_time=None):
        print("Combined audio & video capture stream launched. Press Q to quit.")

        # Format input sources
        input_device = f'{self.video_device}:{self.audio_device}'
        video_shape = f'{self.video_width}x{self.video_height}'

        # Setup stream (with timeout if requested)
        if str(total_processing_time).isnumeric():
            stream = ffmpeg.input(
                input_device, t=total_processing_time,
                format='avfoundation', pixel_format='yuyv422',
                framerate=self.video_fps, s=video_shape
            )
        else:
            stream = ffmpeg.input(
                input_device,
                format='avfoundation', pixel_format='yuyv422',
                framerate=self.video_fps, s=video_shape
            )

        # Setup output segment stream
        stream = ffmpeg.output(
            stream, 'segment-%d.mp4',
            f='segment', segment_time=self.segment_length_s,
            reset_timestamps=1, flags='+global_header',
            sc_threshold=0, g=self.segment_length_s, force_key_frames=f'expr:gte(t, n_forced * {self.segment_length_s})'
        )

        # Run stream continuously
        ffmpeg.run(stream, quiet=False)


class AudioStream():
    def __init__(self, device=1, sample_rate=44100, audio_channels=1):
        self.format = pyaudio.paInt16
        self.rate = sample_rate
        self.chunk = 1024
        self.stream = None
        self.stream_open = False

        self.audio_device = device
        self.audio = pyaudio.PyAudio()
        # self.audio_channels = self.audio.get_device_info_by_host_api_device_index(0, self.audio_device).get('maxInputChannels')
        self.audio_channels = audio_channels

        print(f"     * Audio:")
        print(f"         * Capture device         : {self.audio_device}")
        print(f"         * Input channels         : {self.audio_channels}")

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
            print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        print("Audio thread ended.")

    def kill(self):
        self.stream_open = False
        print("Microphone turned off.")


class VideoStream():
    def __init__(self, device=0, aspect_ratio_x=1280, aspect_ratio_y=720):
        # Define a video capture object
        self.video_device = device
        self.video_stream = cv2.VideoCapture(self.video_device)

        self.video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, aspect_ratio_x)
        self.video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, aspect_ratio_y)

        self.frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.stream_open = False

        # self.width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = aspect_ratio_x
        self.height = aspect_ratio_y

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
                print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

        self.video_stream.release()
        cv2.destroyAllWindows()
        print("\nVideo thread ended.")

    def kill(self):
        self.stream_open = False
        print("Camera turned off.")
