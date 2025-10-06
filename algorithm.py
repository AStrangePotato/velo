import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import mediapipe as mp

def find_smash_frame(video_path):
    try:
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                return 0
            
            audio = video_clip.audio
            audio_waveform = audio.to_soundarray(fps=audio.fps)
            
            if audio_waveform.ndim > 1:
                audio_waveform = audio_waveform.mean(axis=1)

            peak_audio_sample = np.argmax(np.abs(audio_waveform))
            peak_time_sec = peak_audio_sample / audio.fps
            smash_frame = int(peak_time_sec * video_clip.fps)
            return smash_frame
    except Exception as e:
        print(f"Audio processing error for {video_path}: {e}")
        return 0

def process_frame_for_pose(frame, pose_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = pose_model.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr, results