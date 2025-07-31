import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def process_all_videos(frame_root_dir, output_root_dir):
    """
    Process all video folders in a root directory and save only frames with detected hands.

    Args:
        frame_root_dir (str): Root directory containing video folders with frames.
        output_root_dir (str): Root directory to save frames with detected hands.

    Returns:
        None
    """
    try:
        if not os.path.exists(frame_root_dir):
            print(f"Error: Frame root directory {frame_root_dir} does not exist.")
            return

        for video_folder in tqdm(os.listdir(frame_root_dir), desc="Processing videos"):
            video_dir = os.path.join(frame_root_dir, video_folder)
            if not os.path.isdir(video_dir):
                continue

            output_dir = os.path.join(output_root_dir, video_folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Processing video: {video_folder}")

            frame_files = sorted(os.listdir(video_dir))
            count = 0

            for frame_file in frame_files:
                frame_path = os.path.join(video_dir, frame_file)
                if not (frame_file.endswith('.jpg') or frame_file.endswith('.png')):
                    continue

                frame = cv2.imread(frame_path)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Visualize input frame
                cv2.imshow("Input Frame", frame)

                # Detect hands in the frame
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    print(f"Hand landmarks detected in frame: {frame_file}")
                    # Save the frame with detected hands
                    output_path = os.path.join(output_dir, frame_file)
                    cv2.imwrite(output_path, frame)
                    count += 1
                else:
                    print(f"No hand landmarks detected in frame: {frame_file}")

                # Exit on 'q' key press for debugging
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(f"Saved {count} frames with detected hands to {output_dir}")

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing videos in {frame_root_dir}: {e}")

# Example Usage
frame_root_dir = r"C:\Users\bilal\Downloads\fyp_appfinal\fyp_app\frames_output"  # Root directory containing all video folders
output_root_dir = r"C:\Users\bilal\Downloads\fyp_appfinal\fyp_app\filtered_frames_output"  # Root directory for output

process_all_videos(frame_root_dir, output_root_dir)
