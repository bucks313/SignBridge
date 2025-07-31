import os
import shutil
from collections import Counter

src = 'dataset/SL'
dst = 'dataset/SL_top300'
os.makedirs(dst, exist_ok=True)

# Count number of videos in each class
class_counts = Counter()
for cls in os.listdir(src):
    class_dir = os.path.join(src, cls)
    if not os.path.isdir(class_dir):
        continue
    n = len([f for f in os.listdir(class_dir) if f.endswith('.mp4')])
    class_counts[cls] = n

# Get top 300 classes
top300 = [c for c, _ in class_counts.most_common(300)]

# Copy directories
for cls in top300:
    src_dir = os.path.join(src, cls)
    dst_dir = os.path.join(dst, cls)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"Copied {cls} ({class_counts[cls]} videos)")

print(f"\nCreated SL_top300 with {len(top300)} classes.") 