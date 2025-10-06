import cv2
from collections import deque

import config
from calc import track_players, detect_shuttlecock, estimate_shot_trajectory
from visualization import draw_player_bounding_box, draw_shuttlecock_position, draw_trajectory

PLAYER_DETECTOR_MODEL = None
SHUTTLECOCK_MODEL = None

SHUTTLECOCK_TRAIL = deque(maxlen=32)

def process_frame_for_detection(frame):
    """
    This is the core detection and analysis pipeline for a single frame.
    It orchestrates detection, tracking, and visualization.

    Args:
        frame (np.ndarray): The video frame to be processed.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The processed frame with visualizations.
            - dict: A dictionary containing analysis data for this frame.
    """
    analysis_data = {}

    player_boxes = track_players(frame, PLAYER_DETECTOR_MODEL)
    analysis_data['players'] = []
    for i, box in enumerate(player_boxes):
        draw_player_bounding_box(frame, box, player_name=f"Player {i+1}")
        analysis_data['players'].append({'id': i+1, 'bbox': box})

    shuttlecock_pos = detect_shuttlecock(frame, SHUTTLECOCK_MODEL)
    SHUTTLECOCK_TRAIL.append(shuttlecock_pos)
    analysis_data['shuttlecock_position'] = shuttlecock_pos

    if len(SHUTTLECOCK_TRAIL) > 5:
        trajectory_func = estimate_shot_trajectory(list(SHUTTLECOCK_TRAIL), 30)
        if trajectory_func:
            draw_trajectory(frame, trajectory_func, frame.shape[1])
            analysis_data['trajectory_coeffs'] = trajectory_func.coeffs.tolist()

    draw_shuttlecock_position(frame, shuttlecock_pos, trail=list(SHUTTLECOCK_TRAIL))

    return frame, analysis_data
