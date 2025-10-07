import cv2
import mediapipe as mp
import sys
import os

def track_pose(input_file: str):
    """
    Tracks pose in a video, outputs a video with skeleton overlay,
    and returns coordinates of landmarks per frame.
    
    Returns:
        frames_landmarks: List of dictionaries per frame with landmark (x, y, z) positions
    """
    # Prepare output filename
    base_name = os.path.basename(input_file)
    output_file = f"tracked_{base_name}"

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_file}")
        return []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frames_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        frame_landmarks = {}
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save normalized landmark coordinates
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[idx] = {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}

        frames_landmarks.append(frame_landmarks)
        out.write(frame)

    cap.release()
    out.release()
    pose.close()
    print(f"Processing done. Saved to {output_file}")

    return frames_landmarks