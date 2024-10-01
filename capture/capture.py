import os
import signal
import argparse
import sounddevice
from pathlib import Path
from threading import Thread
from collections import deque

from AudioVisualProcessor import AudioVisualProcessor
from AudioVisualStreams import AudioStream, VideoStream, CombinedCaptureStream


if __name__ == '__main__':
    # Recieve input parameters from CLI
    parser = argparse.ArgumentParser(
        prog='capture.py',
        description='Capture audio and video streams from a camera/microphone and split into segments for processing.'
    )

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output/capture/")

    parser.add_argument('-m', '--setup-mode', action='store_true', default=False, help="display video to be captured in setup mode with no capture/processing")
    parser.add_argument('-na', '--no-audio', action='store_false', default=True, help="do not include audio in captured segments")
    parser.add_argument('-nv', '--no-video', action='store_false', default=True, help="do not include video in captured segments")
    parser.add_argument('-s', '--split-av-out', action='store_true', default=False, help="output audio and video in separate files (WAV and MP4)")
    parser.add_argument('-a', '--audio', type=int, default=0, help="index of input audio device")
    parser.add_argument('-v', '--video', type=int, default=0, help="index of input video device")
    parser.add_argument('-o', '--output-path', type=str, default=OUTPUT_DIR, help="directory to output captured video segments to")

    # Decode input parameters to toggle between cameras, microphones, and setup mode.
    args = parser.parse_args()
    audio_device = args.audio
    video_device = args.video
    output_path = args.output_path
    save_av_files = True

    global audio_on, video_on, setup_mode_only, audio, video
    audio_on = args.no_audio
    video_on = args.no_video
    setup_mode_only = args.setup_mode
    split_audio_video = args.split_av_out or not audio_on or not video_on

    print("PRESS 'CTRL+C' TO STOP CAPTURE")
    print(f"\nAudio devices available: \n{sounddevice.query_devices()}", end='\n\n')
    print(f" * Processes:")
    print(f"     * Audio                      : {audio_on}")
    print(f"     * Video                      : {video_on}")
    print(f"     * Save AV segment files      : {save_av_files}")
    print(f"     * Split audio & video tracks : {split_audio_video}",)
    print(f"     * Setup mode (no processing) : {setup_mode_only}", end='\n\n')
    print(f" * Capture setup:")

    # Setup mode
    if setup_mode_only:
        # Set up stream to show content but do no processing
        print()
        video = VideoStream(device=video_device)
        video.launch(display_stream=True)

    # Combined audio & video capture
    elif not split_audio_video:
        # Generate segment output location
        if save_av_files:
            av_save_path = os.path.join(output_path, "segments/")
            Path(av_save_path).mkdir(parents=True, exist_ok=True)

        # Set up and launch combined audio-video stream in a thread
        capture = CombinedCaptureStream(audio_device, video_device, av_save_path)
        audio_frame_queue = deque()
        capture_thread = Thread(target=capture.launch, args=())
        capture_thread.start()

    # Separate audio & video capture
    else:
        # Set up and launch separate audio-video stream threads
        if audio_on:
            # Generate audio output location
            if save_av_files:
                audio_save_path = os.path.join(output_path, "audio/")
                Path(audio_save_path).mkdir(parents=True, exist_ok=True)

            # Launch audio thread
            audio = AudioStream(device=audio_device)
            audio_frame_queue = deque()
            audio_thread = Thread(target=audio.launch, args=(audio_frame_queue,))
            audio_thread.start()

        if video_on:
            # Generate video output location
            if save_av_files:
                video_save_path = os.path.join(output_path, "video/")
                Path(video_save_path).mkdir(parents=True, exist_ok=True)

            # Launch video thread
            video_frame_queue = deque()
            video = VideoStream(device=video_device)
            video_thread = Thread(target=video.launch, args=(video_frame_queue,))
            video_thread.start()

        # Run capture and save av segments to local storage (if requested)
        if audio_on and video_on:
            processor = AudioVisualProcessor(
                video_fps=video.frame_rate, video_shape=(video.width, video.height),
                audio_save_path=audio_save_path, video_save_path=video_save_path
            )

            processor.process(
                audio_module=audio, audio_frames=audio_frame_queue, audio_channels=1,
                video_module=video, video_frames=video_frame_queue,
                checkpoint_files=save_av_files
            )
        elif video_on:
            processor = AudioVisualProcessor(
                video_fps=video.frame_rate, video_shape=(video.width, video.height),
                video_save_path=video_save_path
            )

            processor.process(
                video_module=video, video_frames=video_frame_queue,
                checkpoint_files=save_av_files, audio_on=False
            )
        elif audio_on:
            processor = AudioVisualProcessor(audio_save_path=audio_save_path)
            processor.process(
                audio_module=audio, audio_frames=audio_frame_queue, audio_channels=1,
                checkpoint_files=save_av_files, video_on=False
            )
        else:
            exit(0)
