import cv2
import numpy as np
from pose_helper import track_pose
from audio_analyzer import loudest_sound_time

def side_by_side_video(video1_path, video2_path, output_path="aligned_output.mp4"):
    # Track poses (returns list of frames with landmarks)
    frames1 = track_pose(video1_path)
    frames2 = track_pose(video2_path)

    # Loudest sound times
    t1 = loudest_sound_time(video1_path)
    t2 = loudest_sound_time(video2_path)

    # Open videos to read frames
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    fps = max(fps1, fps2)

    # Resolutions
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w1 + w2
    out_h = max(h1, h2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Convert loudest times to frame indices
    f1_offset = int(t1 * fps1)
    f2_offset = int(t2 * fps2)

    # Pad shorter video at start so loudest sounds align
    pre_pad1 = max(0, f2_offset - f1_offset)
    pre_pad2 = max(0, f1_offset - f2_offset)

    # Read all frames into lists
    vid1_frames = []
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        vid1_frames.append(frame)
    cap1.release()

    vid2_frames = []
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        vid2_frames.append(frame)
    cap2.release()

    # Pad frames with black frames
    black1 = np.zeros((h1, w1, 3), dtype=np.uint8)
    black2 = np.zeros((h2, w2, 3), dtype=np.uint8)
    vid1_frames = [black1]*pre_pad1 + vid1_frames
    vid2_frames = [black2]*pre_pad2 + vid2_frames

    # Make equal length
    max_len = max(len(vid1_frames), len(vid2_frames))
    while len(vid1_frames) < max_len:
        vid1_frames.append(black1)
    while len(vid2_frames) < max_len:
        vid2_frames.append(black2)

    # Write side-by-side video
    for f1, f2 in zip(vid1_frames, vid2_frames):
        # Pad height if needed
        if f1.shape[0] < out_h:
            pad = np.zeros((out_h - f1.shape[0], f1.shape[1], 3), dtype=np.uint8)
            f1 = np.vstack([f1, pad])
        if f2.shape[0] < out_h:
            pad = np.zeros((out_h - f2.shape[0], f2.shape[1], 3), dtype=np.uint8)
            f2 = np.vstack([f2, pad])
        combined = np.hstack([f1, f2])
        out.write(combined)

    out.release()
    print(f"Side-by-side video saved to {output_path}")

if __name__ == "__main__":
    side_by_side_video("input_video.mp4", "reference_video.mp4")
