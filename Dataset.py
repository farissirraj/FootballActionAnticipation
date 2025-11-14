import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import os
import math


class SoccerActionDataset(Dataset):
    def __init__(self, base_path, label_path, num_frames):
        self.base_path = Path(base_path)
        self.label_path = self.base_path / label_path
        self.num_frames = num_frames

        with open(self.label_path, "r") as f:
            self.labels = json.load(f)

        i = 0
        self.samples = []
        for video in self.labels['videos']:
            for j in video['annotations']['observation']:
                if j.get('visibility') == "visible":
                    action_frame = math.floor(int(j['position'])*25/1000) + 1
                    sample = {
                        'path': Path(video['path']).parent,
                        'label': j['label'],
                        'action_frame': action_frame,
                    }

                    self.samples.append(sample)
                    # self.load_frames_around_action(sample)
                    # i += 1
                    # if i == 10: break
            # break

        # self.label_to_idx = {label: idx for idx, label in enumerate(self.labels["labels"])}
        unique_labels = sorted(set(s['label'] for s in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}


    def load_frames_around_action(self, sample):
        clip = Path(sample['path'])
        clip_dir = os.path.join(self.base_path, clip)
        action_frame = sample['action_frame']
        start_frame = max(1, action_frame - 25)
        end_frame = min(750, action_frame + 25)
        # print(start_frame, end_frame, sample['label'], action_frame)

        frames = []
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)

        for frame_idx in frame_indices:
            frame_path = Path(os.path.join(clip_dir, f"frame{frame_idx}.jpg"))
            # print(frame_path)
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)
            else:
                print("Frame does not exist")

        return np.array(frames[:self.num_frames])


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_frames_around_action(sample)
        frames = frames.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std

        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        label = self.label_to_idx[sample['label']]
        
        return frames, label



ds = SoccerActionDataset("./data/720p/train", "Labels-ball.json", num_frames=16)

# Test single sample
frames, label = ds[0]
print(f"Shape: {frames.shape}")  # Expected: torch.Size([3, 16, 112, 112])
print(f"Label: {ds.idx_to_label[label]}")

# Test batch
loader = DataLoader(ds, batch_size=8)
batch = next(iter(loader))
print(f"Batch shape: {batch[0].shape}")  # Expected: torch.Size([8, 3, 16, 112, 112])