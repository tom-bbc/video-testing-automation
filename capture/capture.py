import signal
import argparse
import sounddevice
from pathlib import Path
from threading import Thread
from collections import deque

from AudioVisualProcessor import AudioVisualProcessor
from AudioVisualStreams import AudioStream, VideoStream


def signal_handler(sig, frame):
    print("\nCtrl+C detected. Stopping all AV threads.")
    if video_on: video.kill()
    if audio_on and not setup_mode_only: audio.kill()


if __name__ == '__main__':
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='capture.py',
        description='Capture audio and video streams from a camera/microphone and split into segments for processing.'
    )

    parser.add_argument('-s', '--setup-mode', action='store_true', default=False)
    parser.add_argument('-na', '--no-audio', action='store_false', default=True)
    parser.add_argument('-nv', '--no-video', action='store_false', default=True)
    parser.add_argument('-f', '--save-files', action='store_true', default=False)
    parser.add_argument('-a', '--audio', type=int, default=0)
    parser.add_argument('-v', '--video', type=int, default=0)

    # Decode input parameters to toggle between cameras, microphones, and setup mode.
    args = parser.parse_args()
    audio_device = args.audio
    video_device = args.video
    save_av_files = args.save_files

    global audio_on, video_on, setup_mode_only
    audio_on = args.no_audio
    video_on = args.no_video
    setup_mode_only = args.setup_mode

    print(f"\nAudio devices available: \n{sounddevice.query_devices()}", end='\n\n')
    print(f" * Processes:")
    print(f"     * Audio                      : {audio_on}")
    print(f"     * Video                      : {video_on}")
    print(f"     * Save AV segment files      : {save_av_files}",)
    print(f"     * Setup mode (no processing) : {setup_mode_only}", end='\n\n')
    print(f" * Capture setup:")

    global audio, video

    # Set up the handler for Ctrl+C interrupt
    # signal.signal(signal.SIGINT, signal_handler)

    if setup_mode_only:
        # Set up stream to show content but do no processing
        print()
        video = VideoStream(device=video_device)
        video.launch(display_stream=True)

    else:
        # Create segment save locations
        if save_av_files:
            av_save_path = "output/capture/segments/"
            audio_save_path = "output/capture/audio/"
            video_save_path = "output/capture/video/"
            Path(av_save_path).mkdir(parents=True, exist_ok=True)
            Path(audio_save_path).mkdir(parents=True, exist_ok=True)
            Path(video_save_path).mkdir(parents=True, exist_ok=True)

        # Set up and launch audio-video stream threads
        if audio_on:
            audio = AudioStream(device=audio_device)
            audio_frame_queue = deque()
            audio_thread = Thread(target=audio.launch, args=(audio_frame_queue,))
            audio_thread.start()

        if video_on:
            video_frame_queue = deque()
            video = VideoStream(device=video_device)
            video_thread = Thread(target=video.launch, args=(video_frame_queue,))
            video_thread.start()

        # Check if user wants to run detection algorithms, or just save av segments to disk
        if audio_on and video_on:
            if save_av_files:
                processor = AudioVisualProcessor(
                    video_fps=video.frame_rate, video_shape=(video.width, video.height),
                    audio_save_path=audio_save_path, video_save_path=video_save_path, av_save_path=av_save_path
                )
            else:
                processor = AudioVisualProcessor(
                    video_fps=video.frame_rate, video_shape=(video.width, video.height)
                )

            processor.process(
                audio_module=audio, audio_frames=audio_frame_queue, audio_channels=1,
                video_module=video, video_frames=video_frame_queue,
                checkpoint_files=save_av_files
            )
        elif video_on:
            if save_av_files:
                processor = AudioVisualProcessor(
                    video_fps=video.frame_rate, video_shape=(video.width, video.height),
                    audio_save_path=audio_save_path, video_save_path=video_save_path, av_save_path=av_save_path
                )
            else:
                processor = AudioVisualProcessor(
                    video_fps=video.frame_rate, video_shape=(video.width, video.height)
                )

            processor.process(
                video_module=video, video_frames=video_frame_queue,
                checkpoint_files=save_av_files, audio_on=False, synchronize=False
            )
        elif audio_on:
            processor = AudioVisualProcessor()
            processor.process(
                audio_module=audio, audio_frames=audio_frame_queue, audio_channels=1,
                checkpoint_files=save_av_files, video_on=False, synchronize=False
            )
        else:
            exit(0)
