from essentia.standard import FrameGenerator, GapsDetector, ClickDetector
import numpy as np
import datetime

class AudioDetector():
    def __init__(self):
        self.gaps = []
        self.clicks = []

    def process(self, audio: np.ndarray, start_time=0):
        # Normalise audio to the range [-1, 1]
        normalised_audio = audio / np.max(np.abs(audio))

        gaps = self.audio_gap_detection(normalised_audio, start_time)
        self.gaps.extend(gaps)

        clicks = self.audio_click_detection(normalised_audio, start_time)
        self.clicks.extend(clicks)

        return gaps, clicks

    @staticmethod
    def audio_gap_detection(audio_content, start_time=0):
        """Detection of gaps (silences) in the audio signal"""
        # Parameters
        frame_size = 1024         # frame size used for the analysis
        hop_size = 512            # hop size used for the analysis
        minimum_time = 10         # time of the minimum gap duration [ms]
        prepower_threshold = -38  # prepower threshold [dB]
        silence_threshold = -45   # silence threshold [dB]

        # Detection process
        detected_gap_starts, detected_gap_ends = [], []
        gapDetector = GapsDetector(
            frameSize=frame_size, hopSize=hop_size, minimumTime=minimum_time,
            prepowerThreshold=prepower_threshold, silenceThreshold=silence_threshold
        )

        for frame in FrameGenerator(audio_content, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
            frame_starts, frame_ends = gapDetector(frame)
            detected_gap_starts.extend(frame_starts)
            detected_gap_ends.extend(frame_ends)

        gapDetector.reset()
        detected_gap_starts = np.unique(np.round(detected_gap_starts, decimals=2))
        detected_gap_ends = np.unique(np.round(detected_gap_ends, decimals=2))
        detected_gaps = zip(detected_gap_starts, detected_gap_ends)
        output = []

        if start_time != 0:
            for start, end in detected_gaps:
                output.append((
                    start_time + datetime.timedelta(seconds=float(start)),
                    start_time + datetime.timedelta(seconds=float(end))
                ))

        return output

    @staticmethod
    def audio_click_detection(audio_content, start_time=0):
        """Detection of clicks in the audio signal"""
        # Parameters
        frame_size = 512          # frame size used for the analysis
        hop_size = 256            # hop size used for the analysis

        # Detection process
        detected_clicks = []
        clickDetector = ClickDetector(frameSize=frame_size, hopSize=hop_size)

        for frame in FrameGenerator(audio_content, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
            frame_starts, frame_ends = clickDetector(frame)
            detected_clicks.extend(np.mean([frame_starts, frame_ends], axis=0))

        clickDetector.reset()
        detected_clicks = np.unique(np.round(detected_clicks, decimals=2))
        output = []

        if start_time != 0:
            for click in detected_clicks:
                output.append(start_time + datetime.timedelta(seconds=float(click)))

        return output
