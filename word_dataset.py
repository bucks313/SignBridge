import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class WordSignsDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_augmentations=5):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []
        self.labels = []
        
        print("\nLoading dataset...")
        # Load all videos and create augmented samples
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.mp4'):
                    video_path = os.path.join(class_dir, fname)
                    # Verify video can be opened
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Warning: Could not open video: {video_path}")
                        continue
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    print(f"Loaded {fname} ({total_frames} frames)")
                    
                    # Add original video
                    self.samples.append((video_path, False))  # (path, is_augmented)
                    self.labels.append(self.class_map[class_name])
                    # Add augmented versions
                    for _ in range(num_augmentations):
                        self.samples.append((video_path, True))  # (path, is_augmented)
                        self.labels.append(self.class_map[class_name])
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"Samples per class: {len(self.samples) // len(self.class_names)}")

    def load_video(self, video_path, num_frames=16, is_augmented=False):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video file contains no frames: {video_path}")
            
        # For augmented samples, randomly select a subset of frames
        if is_augmented:
            start_frame = random.randint(0, max(0, total_frames - num_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            
            # Apply basic augmentation if needed
            if is_augmented:
                # Random brightness
                brightness = random.uniform(0.8, 1.2)
                frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
                
                # Random contrast
                contrast = random.uniform(0.8, 1.2)
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
                
                # Random horizontal flip
                if random.random() < 0.5:
                    frame = cv2.flip(frame, 1)
            
            frames.append(frame)
            frame_count += 1
            if len(frames) >= num_frames:
                break
                
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames were extracted from the video: {video_path}")
            
        # Pad with last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
            
        frames = np.array(frames)
        frames = frames.astype(np.float32) / 255.0
        frames = frames.transpose(3, 0, 1, 2)
        
        # Verify frame values are in correct range
        if frames.min() < 0 or frames.max() > 1:
            print(f"Warning: Frame values out of range [{frames.min()}, {frames.max()}]")
            
        return torch.from_numpy(frames).float()

    def __getitem__(self, idx):
        video_path, is_augmented = self.samples[idx]
        label = self.labels[idx]
        clip = self.load_video(video_path, is_augmented=is_augmented)
        
        if self.transform:
            transformed_frames = []
            for t in range(clip.size(1)):
                frame = clip[:, t, :, :]
                frame = self.transform(frame)
                transformed_frames.append(frame)
            clip = torch.stack(transformed_frames, dim=1)
            
        return clip, torch.tensor(label, dtype=torch.long) 