import cv2
import numpy as np
import mediapipe as mp
from algorithm import find_smash_frame, process_frame_for_pose
from visualize import draw_pose_landmarks

def main(user_video_path, ref_video_path, output_video_path):
    pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    user_smash_frame = find_smash_frame(user_video_path)
    ref_smash_frame = find_smash_frame(ref_video_path)

    cap_user = cv2.VideoCapture(user_video_path)
    cap_ref = cv2.VideoCapture(ref_video_path)

    frame_offset = ref_smash_frame - user_smash_frame
    if frame_offset > 0:
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    elif frame_offset < 0:
        cap_user.set(cv2.CAP_PROP_POS_FRAMES, abs(frame_offset))

    output_height = 720
    user_w = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
    user_h = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_w = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_h = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))

    user_aspect_ratio = user_w / user_h
    ref_aspect_ratio = ref_w / ref_h
    output_w_user = int(output_height * user_aspect_ratio)
    output_w_ref = int(output_height * ref_aspect_ratio)
    
    output_width = output_w_user + output_w_ref
    frame_size = (output_width, output_height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, frame_size)

    print("Processing videos...")
    while True:
        ret_user, frame_user = cap_user.read()
        ret_ref, frame_ref = cap_ref.read()

        if not ret_user or not ret_ref:
            break

        frame_user, results_user = process_frame_for_pose(frame_user, pose_model)
        frame_ref, results_ref = process_frame_for_pose(frame_ref, pose_model)

        draw_pose_landmarks(frame_user, results_user)
        draw_pose_landmarks(frame_ref, results_ref)

        resized_user = cv2.resize(frame_user, (output_w_user, output_height))
        resized_ref = cv2.resize(frame_ref, (output_w_ref, output_height))

        combined_frame = np.hstack((resized_user, resized_ref))
        video_writer.write(combined_frame)

    cap_user.release()
    cap_ref.release()
    video_writer.release()
    pose_model.close()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_video_path}")

if __name__ == '__main__':
    main('input_video.mp4', 'sample.mp4', 'output_video.avi')