import cv2
import numpy as np
from pose_helper import track_pose
from audio_analyzer import loudest_sound_time

def get_reference_distance(landmarks):
    distances = []
    for frame in landmarks:
        if 11 in frame and 12 in frame:
            x1, y1 = frame[11]['x'], frame[11]['y']
            x2, y2 = frame[12]['x'], frame[12]['y']
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
    return np.median(distances) if distances else 1.0

def normalize_landmarks(landmarks, target_distance):
    current_distance = get_reference_distance(landmarks)
    scale = target_distance / current_distance if current_distance > 0 else 1.0
    
    normalized = []
    for frame in landmarks:
        norm_frame = {}
        for idx, lm in frame.items():
            norm_frame[idx] = {
                'x': lm['x'] * scale,
                'y': lm['y'] * scale,
                'z': lm['z'] * scale,
                'visibility': lm['visibility']
            }
        normalized.append(norm_frame)
    return normalized

def compute_deviation(coords1, coords2):
    total_deviation = 0
    count = 0
    
    min_len = min(len(coords1), len(coords2))
    for i in range(min_len):
        frame1, frame2 = coords1[i], coords2[i]
        for idx in frame1:
            if idx in frame2 and frame1[idx]['visibility'] > 0.5 and frame2[idx]['visibility'] > 0.5:
                x1, y1 = frame1[idx]['x'], frame1[idx]['y']
                x2, y2 = frame2[idx]['x'], frame2[idx]['y']
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_deviation += dist
                count += 1
    
    return total_deviation / count if count > 0 else 0

def compute_per_joint_deviation(coords1, coords2):
    joint_deviations = {}
    
    min_len = min(len(coords1), len(coords2))
    for i in range(min_len):
        frame1, frame2 = coords1[i], coords2[i]
        for idx in frame1:
            if idx in frame2 and frame1[idx]['visibility'] > 0.5 and frame2[idx]['visibility'] > 0.5:
                x1, y1 = frame1[idx]['x'], frame1[idx]['y']
                x2, y2 = frame2[idx]['x'], frame2[idx]['y']
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if idx not in joint_deviations:
                    joint_deviations[idx] = []
                joint_deviations[idx].append(dist)
    
    avg_deviations = {idx: np.mean(devs) for idx, devs in joint_deviations.items()}
    return avg_deviations

def deviation_to_color(deviation, max_dev=0.3):
    normalized = min(deviation / max_dev, 1.0)
    green = int(255 * (1 - normalized))
    red = int(255 * normalized)
    return (0, green, red)

def draw_colored_pose(frame, landmarks, frame_idx, joint_deviations, original_width, original_height):
    if not landmarks or frame_idx >= len(landmarks):
        return frame
    
    frame_lm = landmarks[frame_idx]
    
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22)
    ]
    
    for idx, lm in frame_lm.items():
        if lm['visibility'] > 0.5:
            x = int(lm['x'] * original_width)
            y = int(lm['y'] * original_height)
            color = deviation_to_color(joint_deviations.get(idx, 0))
            cv2.circle(frame, (x, y), 5, color, -1)
    
    for conn in connections:
        if conn[0] in frame_lm and conn[1] in frame_lm:
            if frame_lm[conn[0]]['visibility'] > 0.5 and frame_lm[conn[1]]['visibility'] > 0.5:
                x1 = int(frame_lm[conn[0]]['x'] * original_width)
                y1 = int(frame_lm[conn[0]]['y'] * original_height)
                x2 = int(frame_lm[conn[1]]['x'] * original_width)
                y2 = int(frame_lm[conn[1]]['y'] * original_height)
                avg_dev = (joint_deviations.get(conn[0], 0) + joint_deviations.get(conn[1], 0)) / 2
                color = deviation_to_color(avg_dev)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    return frame

def side_by_side_tracked(video1_path, video2_path, output_path="aligned_output.mp4", target_height=480):
    cap1_orig = cv2.VideoCapture(video1_path)
    cap2_orig = cv2.VideoCapture(video2_path)
    
    w1_orig = int(cap1_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1_orig = int(cap1_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2_orig = int(cap2_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2_orig = int(cap2_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap1_orig.release()
    cap2_orig.release()
    
    coords_input = track_pose(video1_path)
    coords_reference = track_pose(video2_path)
    
    ref_distance = get_reference_distance(coords_reference)
    coords_input_normalized = normalize_landmarks(coords_input, ref_distance)
    coords_reference_normalized = normalize_landmarks(coords_reference, ref_distance)
    
    joint_deviations = compute_per_joint_deviation(coords_input_normalized, coords_reference_normalized)
    
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
    
    coords_input = [{}] * pre_pad1 + coords_input
    
    black1 = np.zeros_like(frames1[0])
    black2 = np.zeros_like(frames2[0])
    
    frames1 = [black1] * pre_pad1 + frames1
    frames2 = [black2] * pre_pad2 + frames2
    
    max_len = max(len(frames1), len(frames2))
    while len(frames1) < max_len: 
        frames1.append(black1)
        coords_input.append({})
    while len(frames2) < max_len: 
        frames2.append(black2)
    
    h1 = frames1[0].shape[0]
    h2 = frames2[0].shape[0]
    scale_factor2 = h1 / h2
    frames2 = [cv2.resize(f, (int(f.shape[1] * scale_factor2), h1)) for f in frames2]
    
    w1 = frames1[0].shape[1]
    w2 = frames2[0].shape[1]
    h = h1
    
    for i in range(len(frames1)):
        frames1[i] = draw_colored_pose(frames1[i], coords_input, i, joint_deviations, w1_orig, h1_orig)
    
    for i in range(len(frames2)):
        frames2[i] = draw_colored_pose(frames2[i], coords_reference, i, joint_deviations, w2_orig, h2_orig)
    
    out_w = w1 + w2
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))
    
    for f1, f2 in zip(frames1, frames2):
        combined = np.hstack([f1, f2])
        out.write(combined)
    
    out.release()
    
    avg_deviation = compute_deviation(coords_input_normalized, coords_reference_normalized)
    
    print(f"Side-by-side video saved to {output_path}")
    print(f"Average deviation: {avg_deviation:.4f}")
    print("Per-joint deviations:")
    for idx, dev in sorted(joint_deviations.items()):
        print(f"  Joint {idx}: {dev:.4f}")

if __name__ == "__main__":
    side_by_side_tracked("input_video.mp4", "reference_video.mp4")