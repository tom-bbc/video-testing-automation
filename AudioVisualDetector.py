import matplotlib.pyplot as plt
import numpy as np
from time import time as timer
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle


from AudioVisualProcessor import AudioVisualProcessor
from EssentiaAudioDetector import AudioDetector
from MaxVQAVideoDetector import VideoDetector

Object = lambda **kwargs: type("Object", (), kwargs)


class AudioVisualDetector(AudioVisualProcessor):
    def __init__(self, video_downsample_frames=64, device='cpu', *args, **kwargs):
        super(AudioVisualDetector, self).__init__(*args, **kwargs)
        self.audio_detector = AudioDetector()
        self.video_detector = VideoDetector(frames=video_downsample_frames, device=device)

    def process(self,
                audio_module=Object(stream_open=False), audio_frames=[], audio_channels=1,
                video_module=Object(stream_open=False, video_device=None), video_frames=[],
                checkpoint_files=False):

        if audio_module.stream_open:
            print(f"         * Segment size           : {self.audio_buffer_len_f}")
            print(f"         * Overlap size           : {self.audio_overlap_len_f}")

        if video_module.stream_open:
            print(f"     * Video:")
            print(f"         * Capture device         : {video_module.video_device}")
            print(f"         * Frame rate             : {self.video_fps}")
            print(f"         * Segment size           : {self.video_buffer_len_f}")
            print(f"         * Overlap size           : {self.video_overlap_len_f}")

        print(f"\nStart of audio-visual processing")

        while (audio_module.stream_open or video_module.stream_open) or \
            (len(audio_frames) >= self.audio_buffer_len_f) or \
            (len(video_frames) >= self.video_buffer_len_f):

            # Audio processing module
            if len(audio_frames) >= self.audio_buffer_len_f:
                audio_segment = self.collate_audio_frames(audio_frames, audio_channels, self.audio_fps, checkpoint_files)
                results = self.audio_detection(
                    audio_segment,
                    plot=True,
                    time_indexed_audio=True
                )
                self.audio_segment_index += 1

            # Video processing module
            if len(video_frames) >= self.video_buffer_len_f:
                video_segment = self.collate_video_frames(video_frames, checkpoint_files)
                results = self.video_detection(
                    video_segment,
                    plot=True,
                    time_indexed_video=True
                )
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

    def audio_detection(self, audio_content, time_indexed_audio=False, detect_gaps=True, detect_clicks=True, plot=False, start_time=0, end_time=0):
        if time_indexed_audio:
            audio = []
            for time, chunk in audio_content:
                audio = np.append(audio, chunk, axis=1)

            start_time = audio_content[0][0]
            end_time = audio_content[-1][0]
        else:
            audio = audio_content

        detected_audio_gaps, detected_audio_clicks = self.audio_detector.process(
            audio,
            start_time=start_time,
            gap_detection=detect_gaps,
            click_detection=detect_clicks
        )

        print(f"\n * Audio detection (segment {self.audio_segment_index}):")
        print(f"     * Segment time range         : {start_time.strftime('%H:%M:%S.%f')} => {end_time.strftime('%H:%M:%S.%f')}")
        print(f"     * Detected gap times         : {[(s.strftime('%H:%M:%S'), e.strftime('%H:%M:%S')) for s, e in detected_audio_gaps]}")
        print(f"     * Detected click times       : {detected_audio_clicks}")

        # Plot audio signal and any detections
        if plot:
            self.plot_audio(audio, detected_audio_gaps, detected_audio_clicks, start_time, end_time)
            print(f"     * Plot generated             : 'audio-plot-{self.audio_segment_index}.png'")

        print()
        return {"gaps": detected_audio_gaps, "clicks": detected_audio_clicks}

    def plot_audio(self, audio_content, gap_times, click_times, startpoint, endpoint):
        # Setup
        plt.rcParams['agg.path.chunksize'] = 1000
        fig, axs = plt.subplots(1, figsize=(20, 10), tight_layout=True)

        # Form timeline over clip
        time_x = np.linspace(0, 1, len(audio_content[0])) * (endpoint - startpoint) + startpoint
        time_index = np.linspace(0, len(audio_content[0]), len(audio_content[0]))

        # Plot L/R/Mono channels
        for idx, audio_channel in enumerate(audio_content):
            axs.plot(time_index, audio_channel, color='k', alpha=0.5, linewidth=0.5, label=f"Channel {idx}")

        # Plot time range of any audio gaps
        if len(gap_times) > 0:
            for start, end in gap_times:
                approx_gap_start = min(time_x, key=lambda dt: abs(dt - start))
                approx_gap_start_idx = np.where(time_x == approx_gap_start)[0][0]
                approx_gap_end = min(time_x, key=lambda dt: abs(dt - end))
                approx_gap_end_idx = np.where(time_x == approx_gap_end)[0][0]

                line = axs.axvspan(approx_gap_start_idx, approx_gap_end_idx, color='b', alpha=0.3)

            line.set_label('Detected gap')

        # Plot time range of any click artefacts
        if len(click_times) > 0:
            for time in click_times:
                approx_click_time = min(time_x, key=lambda dt: abs(dt - time))
                approx_click_idx = np.where(time_x == approx_click_time)[0][0]
                line = axs.axvline(approx_click_idx, color='r', linewidth=1)

            line.set_label('Detected click')

        axs.set_xticks(time_index[::self.audio_fps])
        axs.set_xticklabels([t.strftime('%H:%M:%S') for t in time_x[::self.audio_fps]], fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel("\nCapture Time (H:M:S)", fontsize=14)
        plt.ylabel("Audio Sample Amplitude", fontsize=14)
        plt.title(f"Audio Defect Detection: Segment {self.audio_segment_index} ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')})) \n", fontsize=18)
        plt.legend(loc=1, fontsize=14)
        fig.savefig(f"output/plots/audio-plot-{self.audio_segment_index}.png")
        plt.close(fig)

    def video_detection(self, video_content, time_indexed_video=False, plot=False, start_time=0, end_time=0, epochs=1):
        if time_indexed_video:
            video = []
            for time, frame in video_content:
                video = np.append(video, frame, axis=1)

            start_time = video_content[0][0]
            end_time = video_content[-1][0]
        else:
            video = video_content

        # MaxVQA AI detection process
        print(f"\n * Video detection (segment {self.video_segment_index}):")

        processing_time_start = timer()
        scores = np.zeros(shape=(epochs,), dtype=object)
        for i in range(epochs):
            score_per_patch = self.video_detector.process(video_content)
            scores[i] = np.array(score_per_patch)

        processing_time_end = timer() - processing_time_start
        score_per_patch = np.mean(scores, axis=0)
        local_scores = np.mean(score_per_patch, axis=0)
        global_scores = np.mean(local_scores, axis=1)
        output = local_scores

        print(f"     * Global VQA scores  : {np.array([f'{i}: {s:.2f}' for i, s in enumerate(global_scores)], dtype=str)}")
        print(f"     * Processing time    : {processing_time_end:.2f}s")

        if plot:
            self.plot_local_vqa(local_scores, start_time, end_time)

        print()
        return output

    def plot_local_vqa(self, vqa_values, startpoint=0, endpoint=0, output_file=''):
        # Metrics
        priority_metrics = [7, 9, 11, 13, 14]
        plot_values = vqa_values[priority_metrics]
        titles = {
            "A": "Sharpness",
            "B": "Noise",
            "C": "Flicker",
            "D": "Compression artefacts",
            "E": "Motion fluency"
        }

        # Timestamps
        plot_with_timestamps = startpoint != 0 and endpoint != 0
        if plot_with_timestamps:
            time_x = np.linspace(0, 1, len(plot_values[0])) * (endpoint - startpoint) + startpoint
            time_index = np.linspace(0, len(plot_values[0]), len(plot_values[0]))

        cycol = cycle(mcolors.TABLEAU_COLORS)
        fig, axes = plt.subplot_mosaic("AB;CD;EE", sharex=True, sharey=True, figsize=(12, 9), tight_layout=True)

        for value_id, (ax_id, title) in enumerate(titles.items()):
            mean_over_video = plot_values[value_id].mean()
            std_over_video = plot_values[value_id].std()

            axes[ax_id].set_title(title)
            axes[ax_id].grid(linewidth=0.2)

            axes[ax_id].axhline(mean_over_video, color='black', ls='--', linewidth=0.5)
            axes[ax_id].axhline(mean_over_video - 2 * std_over_video, color='black', ls='--', linewidth=0.5)

            if plot_with_timestamps:
                axes[ax_id].plot(time_index, plot_values[value_id], linewidth=0.75, color=next(cycol))
            else:
                axes[ax_id].plot(plot_values[value_id], linewidth=0.75, color=next(cycol))

        if plot_with_timestamps:
            fig.suptitle(f"MaxVQA Video Defect Detection: Segment {self.video_segment_index} ({time_x[0].strftime('%H:%M:%S')} => {time_x[-1].strftime('%H:%M:%S')})", fontsize=16)
            fig.supxlabel("Capture Time (H:M:S)")
            num_ticks = round(len(plot_values[0])/10)
            plt.xticks(
                ticks=time_index[::num_ticks],
                labels=[t.strftime('%H:%M:%S') for t in time_x[::num_ticks]]
            )
        else:
            fig.suptitle(f"MaxVQA Video Defect Detection: Segment {self.video_segment_index}", fontsize=16)
            fig.supxlabel("Capture Frame")

        fig.supylabel("Absolute score (0-1, bad-good)")
        plt.yticks([0, 0.25, 0.5, 0.75, 1])

        for ax in fig.get_axes():
            ax.label_outer()

        if output_file == '':
            fig.savefig(f"output/plots/video-plot-{self.video_segment_index}.png")
        else:
            fig.savefig(f"output/plots/{output_file}")

        print(f"     * Plot generated     : {f'video-plot-{self.video_segment_index}.png' if output_file == '' else output_file}")
        plt.close(fig)
