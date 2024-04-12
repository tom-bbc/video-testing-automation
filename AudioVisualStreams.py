import datetime
import cv2
import pyaudio


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
            # print(f"Timestamp counter: {timestamp.strftime('%H:%M:%S.%f')}", end='\r')

        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        print("Audio thread ended.")

    def kill(self):
        self.stream_open = False
        print("Microphone turned off.")


class VideoStream():
    def __init__(self, device=0):
        # Define a video capture object
        self.video_device = device
        self.video_stream = cv2.VideoCapture(self.video_device)
        self.frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        print("Camera turned off.")
