import librosa
import numpy as np
import sys

def loudest_sound_time(video_file: str):
    y, sr = librosa.load(video_file, sr=16000, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]  # fewer frames, faster
    max_idx = np.argmax(rms)
    time_of_max = (max_idx * 1024) / sr
    return time_of_max