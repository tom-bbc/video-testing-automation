import cv2
import time
import queue
import pandas as pd


if __name__ == "__main__":

    # Pick camera source
    camera = 0

    # Define a video capture object
    video_stream = cv2.VideoCapture(camera)

    # Set up frame buffer and clip queue parameters & data structures
    clip_length = 5
    frame_rate = video_stream.get(cv2.CAP_PROP_FPS)
    frames_per_buffer = frame_rate * clip_length

    frame_buffer = pd.DataFrame(columns=['timestamps', 'frames'])
    clip_queue = queue.Queue()
    print()

    # Indefinetely watch camera stream and convert into clips in queue
    while(video_stream.isOpened()):

        print(f"Video timestamp counter: {time.strftime('%Y_%m_%d-%H_%M_%S')}", end='\r')

        # Capture the video frame by frame
        ret, frame = video_stream.read()

        if not ret:
            break

        # Consecutively stack frames into a buffer of `clip_length` second video clips
        if len(frame_buffer) < frames_per_buffer:
            time_indexed_frame = pd.DataFrame({'timestamps': time.strftime("%Y_%m_%d-%H_%M_%S"), 'frames':[frame]})
            frame_buffer = pd.concat([frame_buffer, time_indexed_frame], ignore_index=True)

            time_indexed_frame = time_indexed_frame.head(0)

        # When buffer is full, add it to the queue of clips
        else:
            clip_queue.put(frame_buffer)
            next_clip = clip_queue.get()

            frame_buffer = frame_buffer.head(0)

        # Display frame to user
        # cv2.imshow("WebCam", frame)

        # Press the 'q' button to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video_stream.release()

    # Destroy all the windows
    cv2.destroyAllWindows()
