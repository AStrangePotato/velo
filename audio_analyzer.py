import librosa
import numpy as np

def loudest_sound_time(video_file: str):
    """
    Returns the time (in seconds) of the loudest sound in a video.
    """
    # Load audio from video (librosa can read from ffmpeg-supported files)
    y, sr = librosa.load(video_file, sr=None)  # sr=None preserves original sampling rate

    # Compute short-term energy (RMS)
    rms = librosa.feature.rms(y=y)[0]

    # Find index of maximum RMS value
    max_idx = np.argmax(rms)

    # Convert frame index to time (seconds)
    hop_length = 512  # default in librosa.feature.rms
    time_of_max = (max_idx * hop_length) / sr

    return time_of_max