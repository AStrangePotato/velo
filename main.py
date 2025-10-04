import cv2
import os

# Main application imports
import config
from utils import ensure_dir_exists, save_results
from preprocessing import (
    read_video,
    get_video_properties,
    resize_frame,
    create_background_subtractor,
    subtract_background
)
from detect import process_frame_for_detection
from visualization import display_stats

def main():
    """
    Main function to run the badminton video analysis pipeline.
    """
    # --- Setup ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(config.VIDEO_OUTPUT_PATH)
    ensure_dir_exists(output_dir)

    # --- Video Loading ---
    try:
        cap = read_video(config.VIDEO_INPUT_PATH)
    except IOError as e:
        print(f"Error: {e}")
        return

    width, height, fps = get_video_properties(cap)
    print(f"Input video properties: {width}x{height} @ {fps:.2f} FPS")

    # --- Video Writer Setup ---
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
    out = cv2.VideoWriter(
        config.VIDEO_OUTPUT_PATH,
        fourcc,
        fps,
        (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
    )

    # --- Main Processing Loop ---
    frame_count = 0
    all_analysis_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Preprocessing ---
        # Resize frame for consistent processing
        frame = resize_frame(frame, config.RESIZE_WIDTH, config.RESIZE_HEIGHT)

        # --- Detection and Analysis ---
        # The core logic is encapsulated in process_frame_for_detection
        processed_frame, frame_analysis = process_frame_for_detection(frame.copy())
        frame_analysis['frame_number'] = frame_count
        all_analysis_data.append(frame_analysis)

        # --- Visualization ---
        # Display mock statistics on the frame
        mock_stats = {
            "Rally Time": f"{(frame_count / fps):.1f}s",
            "Last Stroke": "Smash" # This would come from analysis
        }
        display_stats(processed_frame, mock_stats)

        # --- Output ---
        # Write the processed frame to the output video file
        out.write(processed_frame)

        # Display the resulting frame (optional)
        cv2.imshow('Badminton Analysis', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        print(f"Processed frame {frame_count}", end='\r')

    # --- Cleanup ---
    print("\nFinished processing video.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- Save Results ---
    results_to_save = {
        "video_properties": {"width": width, "height": height, "fps": fps},
        "analysis_summary": {"total_frames": frame_count},
        "frame_by_frame_data": all_analysis_data
    }
    save_results(results_to_save, config.RESULTS_JSON_PATH)

if __name__ == '__main__':
    main()