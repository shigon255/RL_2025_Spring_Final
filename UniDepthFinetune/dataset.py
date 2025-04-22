import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiCamDepthDataset(Dataset):
    """
    Loads multi‑cam RGB+depth from .npz files and returns all five views.
    Expects each .npz to contain the keys:
      left_shoulder_rgb,  left_shoulder_depth,
      right_shoulder_rgb, right_shoulder_depth,
      wrist_rgb,          wrist_depth,
      front_rgb,          front_depth
    """
    def __init__(self, npz_dir, transform=None):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        self.transform = transform
        # list of camera names
        self.cams = [
            'left_shoulder',
            'right_shoulder',
            'wrist',
            'front',
        ]
        self.intrinsics = {
        "f": np.array([351.6771208, 351.6771208, 221.70249591, 351.6771208]), # focal length
        "intrinsics": np.array([
            [[351.6771208,   0.,         128.],
             [  0.,         351.6771208, 128.],
             [  0.,           0.,         1.]],
            [[351.6771208,   0.,         128.],
             [  0.,         351.6771208, 128.],
             [  0.,           0.,         1.]],
            [[221.70249591,  0.,         128.],
             [  0.,         221.70249591, 128.],
             [  0.,           0.,         1.]],
            [[351.6771208,   0.,         128.],
             [  0.,         351.6771208, 128.],
             [  0.,           0.,         1.]]
        ])  # Shape: (4, 3, 3)
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        rgb_views = []
        depth_views = []

        for cam in self.cams:
            rgb_np   = data[f'{cam}_rgb']   # (H,W,3) uint8
            depth_np = data[f'{cam}_depth'] # (H,W)   float32

            # to torch and normalize
            rgb_t   = torch.from_numpy(rgb_np).permute(2,0,1).float() / 255.0  # (3,H,W)
            depth_t = torch.from_numpy(depth_np).unsqueeze(0)                 # (1,H,W)

            rgb_views.append(rgb_t)
            depth_views.append(depth_t)

        # stack into (5,3,H,W) and (5,1,H,W)
        rgb_tensor   = torch.stack(rgb_views, dim=0)
        depth_tensor = torch.stack(depth_views, dim=0)

        sample = {'rgb': rgb_tensor, 'depth': depth_tensor, 'intrinsics': self.intrinsics["intrinsics"]}
        if self.transform:
            sample = self.transform(sample)
        return sample