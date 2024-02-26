import cv2
import time
import numpy as np
from threading import Thread
from multiprocessing import Queue


class VideoStream():
    def __init__(self):
        # Pick camera source
        camera = 0

        # Define a video capture object
        self.video_stream = cv2.VideoCapture(camera)

        # Set up frame buffer and clip queue parameters & data structures
        clip_length = 5
        frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.frames_per_buffer = frame_rate * clip_length

    def launch(self, clip_queue):
        # Indefinately watch camera stream and convert into clips in queue
        frame_buffer = {
            'timestamp': [],
            'frame': []
        }

        while(self.video_stream.isOpened()):
            # print(f"Video timestamp counter: {time.strftime('%Y:%m:%d %H:%M:%S')}", end='\r')

            # Capture the video frame by frame
            ret, frame = self.video_stream.read()

            if not ret:
                break

            # Consecutively stack frames into a buffer of `clip_length` second video clips
            if len(frame_buffer['frame']) < self.frames_per_buffer:
                frame_buffer['timestamp'].append(time.strftime("%Y:%m:%d-%H:%M:%S"))
                frame_buffer['frame'].append(frame)

            # When buffer is full, add it to the queue of clips
            else:
                clip_queue.put(frame_buffer.copy())
                frame_buffer['timestamp'] = []
                frame_buffer['frame'] = []

    def kill(self):
        # After the loop release the cap object
        self.video_stream.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

        print("\nStream open: ", self.video_stream.isOpened())


def process_frame_buffer(next_clip):
    # single_frame = next_clip['frame'][-1]
    # cv2.imwrite('output/frame.jpg', single_frame)

    # Detect average brightness
    next_clip['brightness'] = []

    for frame in next_clip['frame']:
        average_brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
        next_clip['brightness'].append(average_brightness)

    print(f"\nTime range: {next_clip['timestamp'][0].split('-')[-1]}-{next_clip['timestamp'][-1].split('-')[-1]}")
    print(f"Average brightness: {np.average(next_clip['brightness']):.2f}")


if __name__ == '__main__':
    # Initialise video stream
    print("Initialising video stream capture...")
    clips = Queue()
    video = VideoStream()

    # Launch video stream in its own thread
    video_thread = Thread(target=video.launch, args=(clips,))
    video_thread.start()
    print("Stream set up. Processing live frames...")

    while True:
        try:
            if not clips.empty():
                next_clip = clips.get()
                process_frame_buffer(next_clip)

        except KeyboardInterrupt:
            video.kill()
            video_thread.join()
            break
