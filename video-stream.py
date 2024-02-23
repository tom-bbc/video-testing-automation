import cv2
import time
import pandas as pd
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
        frame_buffer = pd.DataFrame(columns=['timestamps', 'frames'])

        while(self.video_stream.isOpened()):

            print(f"Video timestamp counter: {time.strftime('%Y_%m_%d-%H_%M_%S')}", end='\r')

            # Capture the video frame by frame
            ret, frame = self.video_stream.read()

            if not ret:
                break

            # Consecutively stack frames into a buffer of `clip_length` second video clips
            if len(frame_buffer) < self.frames_per_buffer:
                time_indexed_frame = pd.DataFrame({
                    'timestamps': time.strftime("%Y_%m_%d-%H_%M_%S"),
                    'frames':[frame]
                })
                frame_buffer = pd.concat(
                    [frame_buffer, time_indexed_frame],
                    ignore_index=True
                )

                time_indexed_frame = time_indexed_frame.head(0)

            # When buffer is full, add it to the queue of clips
            else:
                clip_queue.put(frame_buffer)
                frame_buffer = frame_buffer.head(0)

            # Display frame to user
            # cv2.imshow("WebCam", frame)

    def kill(self):
        # After the loop release the cap object
        self.video_stream.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

        print("\nStream open: ", self.video_stream.isOpened())


def process_frame_buffer(next_clip):
    single_frame = next_clip['frames'].tail(1).to_numpy()[0]
    # print(single_frame)
    cv2.imwrite('output/frame.jpg', single_frame)


if __name__ == '__main__':
    # Initialise video stream
    clips = Queue()
    video = VideoStream()

    # Launch threads
    thread1 = Thread(target=video.launch, args=(clips,))

    # Starting the two threads
    thread1.start()

    while True:
        try:
            if not clips.empty():
                next_clip = clips.get()
                process_frame_buffer(next_clip)

        except KeyboardInterrupt:
            video.kill()
            thread1.join()
            break
