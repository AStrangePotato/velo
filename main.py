import cv2
from preprocess import clean_frame
from detect import RacketTracker

def run():
    in_path = 'input_video.mp4'
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Error: Can't open {in_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = 'swing_analysis.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tracker = RacketTracker(fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean = clean_frame(frame)
        
        new_frame, speed = tracker.find_and_track_racket(frame, clean)
        
        speed_txt = f"Speed: {speed:.2f} px/s"
        peak_txt = f"Peak: {tracker.peak_racket_speed:.2f} px/s"
        
        cv2.putText(new_frame, speed_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(new_frame, peak_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(new_frame)

    print(f"Peak speed: {tracker.peak_racket_speed:.2f} pixels/sec")
    print(f"Output saved to {out_path}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
