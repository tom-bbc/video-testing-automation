import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyaudio
import wave

from AudioVisualProcessor import AudioVisualProcessor
from EssentiaAudioDetector import AudioDetector
from MaxVQAVideoDetector import VideoDetector


Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualDetector(AudioVisualProcessor):
    def __init__(self, *args, **kwargs):
        super(AudioVisualDetector, self).__init__(*args, **kwargs)
        self.audio_detector = AudioDetector()
        # self.video_detector = VideoDetector()

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                audio_gap_detection=True, audio_click_detection=True, checkpoint_files=False):
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
            if len(audio_frames) >= self.audio_buffer_len_f:
                audio_segment = self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps, checkpoint_files)
                self.audio_detection(audio_segment, audio_gap_detection, audio_click_detection)
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                video_segment = self.collate_video_frames(video_frames, checkpoint_files)
                self.video_detection(video_segment)
                self.video_segment_index += 1

        # Save all detection timestamps to CSV database
        if len(self.audio_detector.gaps) > 0:
            detected_gap_timestamps = np.array([(s.strftime('%H:%M:%S.%f'), e.strftime('%H:%M:%S.%f')) for s, e in self.audio_detector.gaps])
            np.savetxt("output/detected_gaps.csv", detected_gap_timestamps, delimiter=",", fmt='%s', header='Timestamp')

        if len(self.audio_detector.clicks) > 0:
            detected_click_timestamps = np.array([t.strftime('%H:%M:%S.%f') for t in self.audio_detector.clicks])
            np.savetxt("output/detected_clicks.csv", detected_click_timestamps, delimiter=",", fmt='%s', header='Timestamp')

        print(f"\nProcessing module ended.")
        print(f"Remaining unprocessed frames: {len(audio_frames)} audio and {len(video_frames)} video \n")

    def collate_audio_frames(self, frame_queue, no_channels=1, sample_rate=44100, save_wav_file=False):
        frame_bytes_buffer = []
        timestamps = []

        # Add main frames in video segment to buffer
        for _ in range(self.audio_buffer_len_f - self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue.popleft()
            timestamps.append(timestamp)
            frame_bytes_buffer.append(frame_bytes)

         # Add overlap frames to buffer
        for i in range(self.audio_overlap_len_f):
            timestamp, frame_bytes = frame_queue[i]
            timestamps.append(timestamp)
            frame_bytes_buffer.append(frame_bytes)

        # Decode all frames from byte representation and arrange into timestamped segment
        frame_buffer = []
        for timestamp, frame_bytes in zip(timestamps, frame_bytes_buffer):
            frame = np.frombuffer(frame_bytes, np.int16).astype(np.float32)

            # Format stereo and mono channel audio into shape (1, 1024) or (2, 1024)
            if len(frame) == 2 * self.chunk_size:
                channel0 = frame[0::2]
                channel1 = frame[1::2]
                frame = np.array([channel0, channel1])
            else:
                frame = np.expand_dims(frame, axis=0)

            frame_buffer.append((timestamp, frame))

        # Save audio data to WAV file for checking later
        if save_wav_file:
            wav_file = wave.open(f'output/data/audio-segment-{self.audio_segment_index}.wav', 'wb')
            wav_file.setnchannels(no_channels)
            wav_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(frame_bytes_buffer))
            wav_file.close()

        return np.array(frame_buffer, dtype=object)

    def collate_video_frames(self, frame_queue, save_mp4_file=False):
        # Setup memory buffer of frames and output video file
        frame_buffer = []
        if save_mp4_file:
            output_file = cv2.VideoWriter(
                f"output/data/video-segment-{self.video_segment_index}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.video_fps,
                self.video_shape
            )

        # Add main frames in video segment to buffer
        for _ in range(self.video_buffer_len_f - self.video_overlap_len_f):
            frame = frame_queue.popleft()
            frame_buffer.append(frame)
            if save_mp4_file: output_file.write(frame[1])

        # Add overlap frames to buffer
        for i in range(self.video_overlap_len_f):
            frame = frame_queue[i]
            frame_buffer.append(frame)
            if save_mp4_file: output_file.write(frame[1])

        if save_mp4_file: output_file.release()

        return frame_buffer

    def audio_detection(self, audio_content, detect_gaps=True, detect_clicks=True, plot=True):
        no_channels = audio_content[0][1].shape[0]
        audio_y = np.array([[]] * no_channels)
        time_x = []

        for time, chunk in audio_content:
            time_x.extend([time] * chunk.shape[1])
            audio_y = np.append(audio_y, chunk, axis=1)

        detected_audio_gaps, detected_audio_clicks = self.audio_detector.process(
            audio_y,
            start_time=audio_content[0][0],
            gap_detection=detect_gaps,
            click_detection=detect_clicks
        )

        print(f"\n * Audio detection (segment {self.audio_segment_index}):")
        print(f"     * Segment time range  : {audio_content[0][0].strftime('%H:%M:%S.%f')[:-4]} => {audio_content[-1][0].strftime('%H:%M:%S.%f')[:-4]}")
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

            # Plot time range of any audio gaps
            if len(detected_audio_gaps) > 0:
                for start, end in detected_audio_gaps:
                    approx_gap_start = min(time_x, key=lambda dt: abs(dt - start))
                    approx_gap_start_idx = time_x.index(approx_gap_start)
                    approx_gap_end = min(time_x, key=lambda dt: abs(dt - end))
                    approx_gap_end_idx = time_x.index(approx_gap_end)

                    line = axs.axvspan(approx_gap_start_idx, approx_gap_end_idx, color='b', alpha=0.3)

                line.set_label('Detected gap')

            # Plot time range of any click artefacts
            if len(detected_audio_clicks) > 0:
                for time in detected_audio_clicks:
                    approx_click_time = min(time_x, key=lambda dt: abs(dt - time))
                    approx_click_idx = time_x.index(approx_click_time)
                    line = axs.axvline(approx_click_idx, color='r', linewidth=1)

                line.set_label('Detected click')

            axs.set_xticks(time_index[::44100])
            axs.set_xticklabels([t.strftime('%H:%M:%S') for t in time_x[::44100]])

            plt.xlabel('Capture Time (H:M:S)')
            plt.ylabel('Audio Sample')
            plt.title(f"Audio Defect Detection: Segment {self.audio_segment_index} ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')})) \n")
            plt.legend(loc=1)
            fig.savefig(f"output/plots/audio-plot-{self.audio_segment_index}.png")
            plt.close(fig)

            print(f"     * Plot generated      : 'audio-plot-{self.audio_segment_index}.png'")

    def video_detection(self, video_content, plot=True):
        # Detect average brightness
        brightness = []
        black_frame_detected = False

        # self.video_detector.process(video_content)

        for time, frame in video_content:
            average_brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
            brightness.append(average_brightness)
            if average_brightness < 10: black_frame_detected = True

        print(f"\n * Video detection (segment {self.video_segment_index}):")
        print(f"     * Segment time range  : {video_content[0][0].strftime('%H:%M:%S.%f')[:-4]} => {video_content[-1][0].strftime('%H:%M:%S.%f')[:-4]}")
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
            fig.savefig(f"output/plots/video-plot-{self.video_segment_index}.png")
            plt.close(fig)

            print(f"     * Plot generated      : 'video-plot-{self.video_segment_index}.png'")
