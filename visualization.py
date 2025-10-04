import cv2
import numpy as np

# This file contains functions for visualizing the analysis on video frames.

def draw_player_bounding_box(frame, player_box, player_name="Player"):
    """
    Draws a bounding box and a label for a player.

    Args:
        frame (np.ndarray): The frame to draw on.
        player_box (tuple): The bounding box (x, y, w, h).
        player_name (str): The name or ID of the player.
    """
    x, y, w, h = player_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue box
    cv2.putText(frame, player_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def draw_shuttlecock_position(frame, shuttlecock_pos, trail=None):
    """
    Draws a circle at the shuttlecock's position and its recent trail.

    Args:
        frame (np.ndarray): The frame to draw on.
        shuttlecock_pos (tuple): The (x, y) coordinates of the shuttlecock.
        trail (list): A list of previous (x, y) coordinates to draw a trail.
    """
    if shuttlecock_pos:
        # Draw the trail
        if trail and len(trail) > 1:
            for i in range(1, len(trail)):
                if trail[i-1] is None or trail[i] is None:
                    continue
                # Decrease thickness for older points
                thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
                cv2.line(frame, trail[i-1], trail[i], (0, 255, 255), thickness) # Yellow trail

        # Draw the current position
        cv2.circle(frame, shuttlecock_pos, 8, (0, 0, 255), -1) # Red circle

def display_stats(frame, stats):
    """
    Displays statistics on the frame, such as player speed or stroke type.

    Args:
        frame (np.ndarray): The frame to draw on.
        stats (dict): A dictionary of stats to display.
                      Example: {"Player 1 Speed": "15 km/h", "Last Stroke": "Smash"}
    """
    y_offset = 40
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40

def draw_trajectory(frame, trajectory_func, width):
    """
    Draws the estimated trajectory of the shuttlecock.

    Args:
        frame (np.ndarray): The frame to draw on.
        trajectory_func (np.poly1d): The polynomial function representing the trajectory.
        width (int): The width of the frame to know the drawing range.
    """
    if trajectory_func is None:
        return
    
    points = []
    for x in range(0, width, 10):
        y = int(trajectory_func(x))
        if 0 < y < frame.shape[0]:
            points.append((x, y))
    
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)
