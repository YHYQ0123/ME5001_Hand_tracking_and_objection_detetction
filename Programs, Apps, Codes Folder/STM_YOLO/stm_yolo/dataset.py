import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MultiViewVideoDataset(Dataset):
    def __init__(self, frames_stage_dir, labels_stage_dir, sequence_length=5, transform=None, video_folders=None):
        self.frames_stage_dir = frames_stage_dir
        self.labels_stage_dir = labels_stage_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_folders_to_use = video_folders
        self.samples = []
        self.label_cache = {}
        self._prepare_samples()

    def _prepare_samples(self):
        for video_folder in sorted(self.video_folders_to_use):
            frame_dir = os.path.join(self.frames_stage_dir, video_folder)
            label_dir = os.path.join(self.labels_stage_dir, video_folder)
            label_file = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
            if not label_file:
                continue
            label_path = os.path.join(label_dir, label_file[0])
            labels = pd.read_csv(label_path, header=None)[0].tolist()
            self.label_cache[video_folder] = labels

            kinect_dir = os.path.join(frame_dir, 'kinect')
            num_frames = len([f for f in os.listdir(kinect_dir) if f.endswith('.jpg')])
            if num_frames < self.sequence_length:
                continue

            for end_idx in range(self.sequence_length - 1, num_frames):
                self.samples.append({
                    'video': video_folder,
                    'frame_range': list(range(end_idx - self.sequence_length + 1, end_idx + 1)),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = sample['video']
        frames = sample['frame_range']
        label = self.label_cache[video][frames[-1]]

        view_tensors = []
        for view in ['left', 'right', 'kinect']:
            imgs = []
            for i in frames:
                img_path = os.path.join(self.frames_stage_dir, video, view, f"{str(i).zfill(5)}.jpg")
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            view_tensors.append(torch.stack(imgs))  # [T, C, H, W]

        return view_tensors[0], view_tensors[1], view_tensors[2], torch.tensor(label, dtype=torch.long)
