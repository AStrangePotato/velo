import cv2
import numpy as np

class RacketTracker:
    def __init__(self, video_fps):
        self.previous_racket_position = None
        self.frames_per_second = video_fps
        self.peak_racket_speed = 0
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def find_and_track_racket(self, frame, preprocessed_frame):
        foreground_mask = self.background_subtractor.apply(preprocessed_frame)
        _, foreground_mask = cv2.threshold(foreground_mask, 244, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(foreground_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        racket_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5.0:
                    if area > max_area:
                        max_area = area
                        racket_contour = contour

        current_racket_position = None
        current_racket_speed = 0

        if racket_contour is not None:
            moments = cv2.moments(racket_contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                current_racket_position = (center_x, center_y)

                if self.previous_racket_position is not None:
                    pixels_moved = np.linalg.norm(np.array(current_racket_position) - np.array(self.previous_racket_position))
                    current_racket_speed = pixels_moved * self.frames_per_second

                    if current_racket_speed > self.peak_racket_speed:
                        self.peak_racket_speed = current_racket_speed
                
                self.previous_racket_position = current_racket_position
                
                cv2.drawContours(frame, [racket_contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, current_racket_position, 5, (0, 0, 255), -1)

        return frame, current_racket_speed
