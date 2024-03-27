import cv2
import datetime
import math
import numpy as np
from threading import Thread
from collections import deque


class CameraStream():
    def __init__(self, camera=0):
        # Define a video capture object
        self.video_stream = cv2.VideoCapture(camera)
        self.frame_rate = self.video_stream.get(cv2.CAP_PROP_FPS)

    def launch(self, frame_queue=None, display_stream=False):
        # Show the video stream (no processing)
        if display_stream:
            while self.video_stream.isOpened():
                _, frame = self.video_stream.read()
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
                cv2.imshow('Video Stream', frame)
                print(f"Video timestamp counter: {timestamp}", end='\r')

                # esc to quit
                if cv2.waitKey(1) == 27:
                    self.kill()
                    break

        # Conduct video processing (save frames to queue instead of displaying)
        else:
            # Flush initial two black frames
            self.video_stream.read()
            self.video_stream.read()

            while self.video_stream.isOpened():
                # Capture the video frame by frame
                _, frame = self.video_stream.read()
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
                frame_queue.append((timestamp, frame))
                print(f"Video timestamp counter: {timestamp}", end='\r')

    def kill(self):
        # After the loop release the cap object
        self.video_stream.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

        print("\nStream open: ", self.video_stream.isOpened())


def video_processing(video_frames, frames_per_second, buffer_length_secs=10, overlap_length_secs=2):
    # Processing frame buffer parameters
    buffer_length_frames = math.ceil(frames_per_second * buffer_length_secs)
    overlap_length_frames = math.ceil(frames_per_second * overlap_length_secs)
    segment_index = 0

    print(f"     * Video segments size        : {buffer_length_frames}")
    print(f"     * Overlapping frames         : {overlap_length_frames}", end='\n\n')

    while True:
        try:
            if len(video_frames) > buffer_length_frames:
                video_segment = collate_frames(video_frames, buffer_length_frames, overlap_length_frames, segment_index)
                detection(video_segment, segment_index)
                segment_index += 1

        except KeyboardInterrupt:
            video.kill()
            video_thread.join()
            break


def collate_frames(frame_queue, frames_per_buffer, overlap_frames, index=0):
    print(f"\n\n * New video segment ({index}):")
    print(f"     * Input queue length  : {len(frame_queue)}")

    # Add main frames in video segment to buffer
    frame_buffer = []
    for _ in range(frames_per_buffer - overlap_frames):
        frame_buffer.append(frame_queue.popleft())

    # Add overlap frames to buffer
    frame_buffer.extend([frame_queue[i] for i in range(overlap_frames)])

    print(f"     * Segment frame count : {len(frame_buffer)}")
    print(f"     * Segment time range  : {frame_buffer[0][0]} => {frame_buffer[-1][0]}")
    print(f"     * Output queue length : {len(frame_queue)}", end='\n\n')

    return frame_buffer


def detection(video, index=0):
    # Detect average brightness
    brightness = []
    black_frame_detected = False

    for time, frame in video:
        average_brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
        brightness.append(average_brightness)
        if average_brightness < 10: black_frame_detected = True

    print(f"\n\n * Video detection ({index}):")
    print(f"     * Average brightness  : {np.average(brightness):.2f}")
    print(f"     * Black frame detected: {black_frame_detected}", end='')


if __name__ == '__main__':
    # Switch between setup video stream & processing stream
    camera_device = 0
    processing_module = True
    video = CameraStream(camera=camera_device)
    video_fps = video.frame_rate

    print(f"\n * Parameters:")
    print(f"     * Process video (no display) : {processing_module}")
    print(f"     * Camera capture source      : {camera_device}")
    print(f"     * Capture frame rate         : {video_fps}")

    if not processing_module:
        # Set up stream to show content but do no processing
        print()
        video.launch(display_stream=True)
    else:
        # Initialise video stream
        video_frame_queue = deque()

        # Set up and launch video stream thread and processing thread
        video_thread = Thread(target=video.launch, args=(video_frame_queue,))
        processing_thread = Thread(target=video_processing, args=(video_frame_queue, video_fps,))

        video_thread.start()
        processing_thread.start()
