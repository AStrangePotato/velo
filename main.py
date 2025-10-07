import cv2
import numpy as np
from pose_helper import track_pose
from audio_analyzer import loudest_sound_time

def side_by_side_tracked(video1_path, video2_path, output_path="aligned_output.mp4", target_height=480):
    # Track poses and generate skeleton-overlaid videos
    track_pose(video1_path)
    track_pose(video2_path)

    tracked1 = f"tracked_{video1_path}"
    tracked2 = f"tracked_{video2_path}"

    # Loudest sound times
    t1 = loudest_sound_time(video1_path)
    t2 = loudest_sound_time(video2_path)

    # Open videos
    cap1 = cv2.VideoCapture(tracked1)
    cap2 = cv2.VideoCapture(tracked2)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = max(fps1, fps2)

    # Read frames
    frames1, frames2 = [], []
    while True:
        ret, frame = cap1.read()
        if not ret: break
        frames1.append(frame)
    cap1.release()

    while True:
        ret, frame = cap2.read()
        if not ret: break
        frames2.append(frame)
    cap2.release()

    # Convert loudest times to frame indices
    f1_offset = int(t1 * fps1)
    f2_offset = int(t2 * fps2)

    # Pad start frames so loudest sounds align
    pre_pad1 = max(0, f2_offset - f1_offset)
    pre_pad2 = max(0, f1_offset - f2_offset)
    black1 = np.zeros_like(frames1[0])
    black2 = np.zeros_like(frames2[0])
    frames1 = [black1]*pre_pad1 + frames1
    frames2 = [black2]*pre_pad2 + frames2

    # Make lengths equal
    max_len = max(len(frames1), len(frames2))
    while len(frames1) < max_len: frames1.append(black1)
    while len(frames2) < max_len: frames2.append(black2)

    # Scale both videos to target_height
    def scale_frame(frame, target_h):
        h, w = frame.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        return cv2.resize(frame, (new_w, target_h))

    frames1 = [scale_frame(f, target_height) for f in frames1]
    frames2 = [scale_frame(f, target_height) for f in frames2]

    # Output video dimensions
    out_h = target_height
    out_w = frames1[0].shape[1] + frames2[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Write side-by-side frames
    for f1, f2 in zip(frames1, frames2):
        combined = np.hstack([f1, f2])
        out.write(combined)

    out.release()
    print(f"Side-by-side video saved to {output_path}")

if __name__ == "__main__":
    side_by_side_tracked("input_video.mp4", "reference_video.mp4")
