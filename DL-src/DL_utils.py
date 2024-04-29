import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class VideoFrameDataset(Dataset):
    def __init__(self, directory, encoder, transform=None):
        super().__init__()
        self.videos = []  # List to hold lists of frame paths for each video
        self.encoder = encoder
        self.window_size = 11
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.transform = transform if transform is not None else transforms.ToTensor()

        # Iterate over each folder
        for video_dir in sorted(os.listdir(directory)):
            video_path = os.path.join(directory, video_dir)
            if os.path.isdir(video_path):
                # Collect all frame file paths
                image_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.png')],key=lambda x: int(x[:-4].split('_')[-1]))
                mask_file = os.path.join(video_path, 'mask.npy')
                if os.path.exists(mask_file):
                    masks = np.load(mask_file)
                    self.videos.append([image_files, masks])
                else:
                    self.videos.append([image_files, None])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_frames, masks = self.videos[idx]
        sequences = []

        # Generate all sequences from 0-10 up to 10-21
        for start in range(self.window_size):  # This creates sequences 0-10, 1-11, ..., 10-21
            frame_sequence = video_frames[start:start+self.window_size]  # Get 11 frames starting from 'start'
            images = self.transform([Image.open(frame) for frame in frame_sequence])
            
            # Encode the frames with the provided encoder
            tensor = torch.stack(images)

            sequences.append(tensor.squeeze(0))
        stacked_sequences = torch.stack(sequences)
        
        if masks is not None:
            return stacked_sequences, masks

        return stacked_sequences, 0
