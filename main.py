import cv2
import numpy as np
from pose_helper import track_pose
from audio_analyzer import loudest_sound_time
from swing_compare import compare_swings, scale_landmarks

def side_by_side_tracked(video1_path, video2_path, output_path="aligned_output.mp4", target_height=480):

    coords_input = track_pose(video1_path)
    coords_reference = track_pose(video2_path)

    tracked1 = f"tracked_{video1_path}"
    tracked2 = f"tracked_{video2_path}"

    t1 = loudest_sound_time(video1_path)
    t2 = loudest_sound_time(video2_path)

    cap1 = cv2.VideoCapture(tracked1)
    cap2 = cv2.VideoCapture(tracked2)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = max(fps1, fps2)

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

    f1_offset = int(t1 * fps1)
    f2_offset = int(t2 * fps2)

    pre_pad1 = max(0, f2_offset - f1_offset)
    pre_pad2 = max(0, f1_offset - f2_offset)

    black1 = np.zeros_like(frames1[0])
    black2 = np.zeros_like(frames2[0])

    frames1 = [black1]*pre_pad1 + frames1
    frames2 = [black2]*pre_pad2 + frames2

    max_len = max(len(frames1), len(frames2))
    while len(frames1) < max_len: frames1.append(black1)
    while len(frames2) < max_len: frames2.append(black2)

    h1 = frames1[0].shape[0]
    h2 = frames2[0].shape[0]
    scale_factor2 = h1 / h2
    frames2 = [cv2.resize(f, (int(f.shape[1]*scale_factor2), h1)) for f in frames2]

    cap2 = cv2.VideoCapture(video2_path)
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2_orig = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap2.release()

    coords_reference = scale_landmarks(coords_reference, (w2, h2_orig), (frames2[0].shape[1], h1))

    out_w = frames1[0].shape[1] + frames2[0].shape[1]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h1))

    for f1, f2 in zip(frames1, frames2):
        combined = np.hstack([f1, f2])
        out.write(combined)

    out.release()

    _, _, deviation = compare_swings(coords_input, coords_reference)

    print(f"Side-by-side video saved to {output_path}")
    print("deviation:", deviation)

if __name__ == "__main__":
    side_by_side_tracked("input_video.mp4", "reference_video.mp4")
