import numpy as np

def scale_landmarks(landmarks_list, original_size, target_size):
    ow, oh = original_size
    tw, th = target_size
    sx, sy = tw / ow, th / oh

    scaled_list = []
    for frame_landmarks in landmarks_list:
        scaled_frame = {}
        for idx, lm in frame_landmarks.items():
            scaled_frame[idx] = {
                'x': lm['x'] * sx,
                'y': lm['y'] * sy,
                'z': lm['z'],
                'visibility': lm['visibility']
            }
        scaled_list.append(scaled_frame)
    return scaled_list

def compare_swings(ref_landmarks, test_landmarks):
    """
    Compare two swings using landmark coordinates.
    Deviation is calculated after aligning frames at the right shoulder (landmark 12).
    Returns deviations per frame, average per frame, and overall deviation.
    """
    min_frames = min(len(ref_landmarks), len(test_landmarks))
    deviations = []

    for i in range(min_frames):
        frame_ref = ref_landmarks[i]
        frame_test = test_landmarks[i]
        frame_dev = []

        if 12 not in frame_ref or 12 not in frame_test:
            deviations.append([0])
            continue

        ref_shoulder = frame_ref[12]
        test_shoulder = frame_test[12]

        for idx in frame_ref.keys():
            if idx in frame_test:
                dx = (frame_ref[idx]['x'] - ref_shoulder['x']) - (frame_test[idx]['x'] - test_shoulder['x'])
                dy = (frame_ref[idx]['y'] - ref_shoulder['y']) - (frame_test[idx]['y'] - test_shoulder['y'])
                dz = (frame_ref[idx]['z'] - ref_shoulder['z']) - (frame_test[idx]['z'] - test_shoulder['z'])
                frame_dev.append(np.sqrt(dx**2 + dy**2 + dz**2))

        deviations.append(frame_dev)

    avg_per_frame = [np.mean(f) if f else 0 for f in deviations]
    overall_deviation = np.mean(avg_per_frame)

    return deviations, avg_per_frame, overall_deviation
