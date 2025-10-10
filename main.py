import cv2
import numpy as np
import mediapipe as mp
import os
from audio_analyzer import loudest_sound_time
from pose_helper import track_pose

def get_pose_center_and_scale(frame_landmarks):
    if not frame_landmarks:
        return None, None

    # Use hips and shoulders for a stable center and scale
    left_hip = frame_landmarks.get(23)
    right_hip = frame_landmarks.get(24)
    left_shoulder = frame_landmarks.get(11)
    right_shoulder = frame_landmarks.get(12)

    if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
        return None, None

    if any(lm['visibility'] < 0.5 for lm in [left_hip, right_hip, left_shoulder, right_shoulder]):
        return None, None

    hip_x = (left_hip['x'] + right_hip['x']) / 2
    hip_y = (left_hip['y'] + right_hip['y']) / 2
    shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
    shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
    
    center = np.array([hip_x, hip_y])
    
    # Scale based on torso height
    scale = np.sqrt((shoulder_x - hip_x)**2 + (shoulder_y - hip_y)**2)
    
    return center, scale

def normalize_and_center_landmarks(landmarks_data):
    normalized_landmarks = []
    for frame_landmarks in landmarks_data:
        center, scale = get_pose_center_and_scale(frame_landmarks)
        norm_frame = {}
        if center is not None and scale > 1e-6:
            for idx, lm in frame_landmarks.items():
                norm_x = (lm['x'] - center[0]) / scale
                norm_y = (lm['y'] - center[1]) / scale
                # Z is not used for 2D comparison but kept for data integrity
                norm_z = (lm['z']) / scale 
                norm_frame[idx] = {
                    'x': norm_x, 'y': norm_y, 'z': norm_z, 
                    'visibility': lm['visibility']
                }
        normalized_landmarks.append(norm_frame)
    return normalized_landmarks

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

def deviation_to_color(deviation, max_dev=0.5):
    normalized = min(deviation / max_dev, 1.0)
    green = int(255 * (1 - normalized))
    red = int(255 * normalized)
    return (0, green, red)

def draw_skeleton(frame, frame_landmarks, joint_deviations, is_overlay=False, ref_center=None, ref_scale=None):
    if not frame_landmarks:
        return frame
        
    h, w, _ = frame.shape
    
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]

    points_to_draw = {}
    for idx, lm in frame_landmarks.items():
        if lm['visibility'] < 0.5:
            continue

        if is_overlay:
            # Transform normalized coordinates to reference frame's space
            x = int((lm['x'] * ref_scale + ref_center[0]) * w)
            y = int((lm['y'] * ref_scale + ref_center[1]) * h)
        else:
            # Use raw coordinates
            x = int(lm['x'] * w)
            y = int(lm['y'] * h)
            
        color = deviation_to_color(joint_deviations.get(idx, 0))
        points_to_draw[idx] = ((x, y), color)

    for p1_idx, p2_idx in connections:
        if p1_idx in points_to_draw and p2_idx in points_to_draw:
            pt1, color1 = points_to_draw[p1_idx]
            pt2, color2 = points_to_draw[p2_idx]
            avg_dev = (joint_deviations.get(p1_idx, 0) + joint_deviations.get(p2_idx, 0)) / 2
            line_color = deviation_to_color(avg_dev)
            cv2.line(frame, pt1, pt2, line_color, 2)

    for idx, (pt, color) in points_to_draw.items():
        cv2.circle(frame, pt, 5, color, -1)
        
    return frame


def side_by_side_tracked(video1_path, video2_path, output_path="aligned_output.mp4", target_height=720):
    coords_input = track_pose(video1_path)
    coords_reference = track_pose(video2_path)
    
    coords_input_normalized = normalize_and_center_landmarks(coords_input)
    coords_reference_normalized = normalize_and_center_landmarks(coords_reference)

    joint_deviations = compute_per_joint_deviation(coords_input_normalized, coords_reference_normalized)
    
    t1 = loudest_sound_time(video1_path)
    t2 = loudest_sound_time(video2_path)
    
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = max(fps1, fps2)
    
    f1_offset = int(t1 * fps1)
    f2_offset = int(t2 * fps2)
    
    pre_pad1 = max(0, f2_offset - f1_offset)
    pre_pad2 = max(0, f1_offset - f2_offset)
    
    coords_input_padded = ([{}] * pre_pad1) + coords_input
    coords_reference_padded = ([{}] * pre_pad2) + coords_reference
    coords_input_norm_padded = ([{}] * pre_pad1) + coords_input_normalized

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames1, frames2 = [], []
    while True:
        ret, frame = cap1.read(); 
        if not ret: break
        frames1.append(frame)
    while True:
        ret, frame = cap2.read(); 
        if not ret: break
        frames2.append(frame)

    cap1.release()
    cap2.release()
    
    h1, w1, _ = frames1[0].shape
    h2, w2, _ = frames2[0].shape
    
    aspect1 = w1 / h1
    aspect2 = w2 / h2
    
    new_w1 = int(target_height * aspect1)
    new_w2 = int(target_height * aspect2)
    
    frames1 = [cv2.resize(f, (new_w1, target_height)) for f in frames1]
    frames2 = [cv2.resize(f, (new_w2, target_height)) for f in frames2]
    
    black1 = np.zeros((target_height, new_w1, 3), dtype=np.uint8)
    black2 = np.zeros((target_height, new_w2, 3), dtype=np.uint8)

    frames1_padded = ([black1] * pre_pad1) + frames1
    frames2_padded = ([black2] * pre_pad2) + frames2

    max_len = max(len(frames1_padded), len(frames2_padded))
    while len(frames1_padded) < max_len: frames1_padded.append(black1)
    while len(frames2_padded) < max_len: frames2_padded.append(black2)
    while len(coords_input_padded) < max_len: coords_input_padded.append({})
    while len(coords_reference_padded) < max_len: coords_reference_padded.append({})
    while len(coords_input_norm_padded) < max_len: coords_input_norm_padded.append({})

    output_frames = []
    for i in range(max_len):
        f1 = frames1_padded[i]
        f2 = frames2_padded[i]

        lm1_raw = coords_input_padded[i]
        lm2_raw = coords_reference_padded[i]
        lm1_norm = coords_input_norm_padded[i]

        f1_out = draw_skeleton(f1.copy(), lm1_raw, joint_deviations)

        f2_overlay = f2.copy()
        ref_center, ref_scale = get_pose_center_and_scale(lm2_raw)

        if ref_center is not None and ref_scale is not None:
            # Draw reference skeleton in white
            f2_overlay = draw_skeleton(f2_overlay, lm2_raw, {}) 
            # Draw input skeleton overlayed and colored by deviation
            f2_overlay = draw_skeleton(f2_overlay, lm1_norm, joint_deviations, is_overlay=True, ref_center=ref_center, ref_scale=ref_scale)

        combined = np.hstack([f1_out, f2_overlay])
        output_frames.append(combined)

    out_h, out_w, _ = output_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    avg_deviation = compute_deviation(coords_input_normalized, coords_reference_normalized)
    
    print(f"Side-by-side video saved to {output_path}")
    print(f"Average deviation score: {avg_deviation:.4f}")
    print("Per-joint deviations:")
    for idx, dev in sorted(joint_deviations.items()):
        print(f"  Joint {idx}: {dev:.4f}")

if __name__ == "__main__":
    side_by_side_tracked("input_video.mp4", "reference_video.mp4")
    