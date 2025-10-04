import cv2
import numpy as np

def read_video(video_path):
    """
    Reads a video file and returns a VideoCapture object.

    Args:
        video_path (str): The path to the video file.

    Returns:
        cv2.VideoCapture: The VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    return cap

def get_video_properties(cap):
    """
    Gets properties of a video.

    Args:
        cap (cv2.VideoCapture): The VideoCapture object.

    Returns:
        tuple: A tuple containing frame width, frame height, and fps.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, fps

def convert_to_grayscale(frame):
    """
    Converts a frame to grayscale.

    Args:
        frame (np.ndarray): The input frame.

    Returns:
        np.ndarray: The grayscale frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def resize_frame(frame, width, height):
    """
    Resizes a frame to a specified width and height.

    Args:
        frame (np.ndarray): The input frame.
        width (int): The target width.
        height (int): The target height.

    Returns:
        np.ndarray: The resized frame.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def denoise_frame(frame):
    """
    Applies Gaussian blur to denoise a frame.

    Args:
        frame (np.ndarray): The input frame.

    Returns:
        np.ndarray: The denoised frame.
    """
    return cv2.GaussianBlur(frame, (5, 5), 0)

def subtract_background(frame, bg_subtractor):
    """
    Performs background subtraction on a frame.

    Args:
        frame (np.ndarray): The input frame.
        bg_subtractor: The background subtractor object (e.g., from cv2.createBackgroundSubtractorMOG2).

    Returns:
        np.ndarray: The foreground mask.
    """
    if bg_subtractor is None:
        raise ValueError("Background subtractor is not initialized.")
    return bg_subtractor.apply(frame)

def create_background_subtractor(history=500, var_threshold=16, detect_shadows=True):
    """
    Creates a background subtractor object.

    Args:
        history (int): The number of last frames that affect the background model.
        var_threshold (int): The variance threshold for the pixel-model match.
        detect_shadows (bool): If true, the algorithm will detect shadows and mark them.

    Returns:
        cv2.BackgroundSubtractorMOG2: The background subtractor object.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )
