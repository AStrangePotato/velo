import cv2

def clean_frame(frame):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)
    return blurred_frame
