import os
import cv2

root_dir = "dataset/SL"
corrupted = []

for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for video_name in os.listdir(class_dir):
        if not video_name.endswith('.mp4'):
            continue
        video_path = os.path.join(class_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Corrupted (cannot open): {video_path}")
            corrupted.append(video_path)
            try:
                os.remove(video_path)
                print(f"Deleted: {video_path}")
            except Exception as e:
                print(f"Failed to delete: {video_path} ({e})")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f"Corrupted (0 frames): {video_path}")
            corrupted.append(video_path)
            try:
                os.remove(video_path)
                print(f"Deleted: {video_path}")
            except Exception as e:
                print(f"Failed to delete: {video_path} ({e})")
            cap.release()
            continue
        corrupted_flag = False
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print(f"Corrupted (cannot read frame {i}): {video_path}")
                corrupted_flag = True
                break
        cap.release()
        if corrupted_flag:
            corrupted.append(video_path)
            try:
                os.remove(video_path)
                print(f"Deleted: {video_path}")
            except Exception as e:
                print(f"Failed to delete: {video_path} ({e})")

print(f"\nTotal corrupted videos deleted: {len(corrupted)}") 