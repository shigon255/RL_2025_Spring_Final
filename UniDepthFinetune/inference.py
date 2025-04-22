import einops
import glob
import matplotlib.pyplot as plt
from model import UniDepthV2Finetune
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from unidepth.utils.camera import Pinhole as unidepth_Pinhole



def main():
    # Hyperparams
    npz_dir    = '/project2/yehhh/datasets/RLBench/sim-depth'
    batch_size = 10
    epochs     = 15
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp        = 'finetune_15'
    log_dir    = f'runs/{exp}'

    model = UniDepthV2Finetune(device, correction_head_weights='runs/finetune_20_lr-2schedule_loss1050/correction_model_5.pth')

    model.pretrained.eval()
    model.correction_head.eval()

    files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
    data = np.load(files[669])
    cams = [
        'left_shoulder',
        'right_shoulder',
        'wrist',
        'front',
        ]
    rgb_views = []
    depth_views = []

    for cam in cams:
        rgb_np   = data[f'{cam}_rgb']   # (H,W,3) uint8
        depth_np = data[f'{cam}_depth'] # (H,W)   float32

        # to torch and normalize
        rgb_t   = torch.from_numpy(rgb_np).permute(2,0,1).float() / 255.0  # (3,H,W)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0)                 # (1,H,W)

        rgb_views.append(rgb_t)
        depth_views.append(depth_t)

    # stack into (1,4,3,H,W) and (1,4,1,H,W)
    rgb_tensor   = torch.stack(rgb_views, dim=0)[None].to(device)
    depth_tensor = torch.stack(depth_views, dim=0)[None].to(device)
    intrinsics = torch.tensor([
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
    ], device=device)[None]  # Shape: (1, 4, 3, 3)
    

    
    with torch.no_grad():
        dpreds = model(rgb_tensor, intrinsics)       # (B,4,1,H,W)

        dpreds_pretrained = torch.zeros(dpreds.shape, device=device)

        # loop over batch and view
        for b in range(dpreds.size(0)):
            for v in range(dpreds.size(1)):
                # single view rgb: (1,3,H,W)
                rgb_bv = rgb_tensor[b, v:v+1] * 255  

                K_bv = intrinsics[b, v]                # (3,3)
                camera = unidepth_Pinhole(K=K_bv.unsqueeze(0))

                out = model.pretrained.infer(rgb_bv, camera=camera)
                d_bv = out['depth'].unsqueeze(1)           # (1,1,H,W)

                dpreds_pretrained[b, v] = d_bv

    mae = torch.abs(dpreds - depth_tensor).mean().item()
    print(f'MAE: {mae:.4f}')
    
    gt_all   = depth_tensor[0]  # (5*1*H*W,)
    pred_all = dpreds[0]

    gt_min, gt_max = gt_all.min().item(), gt_all.max().item()
    gt_mean, gt_std = gt_all.mean().item(), gt_all.std().item()

    pr_min, pr_max = pred_all.min().item(), pred_all.max().item()
    pr_mean, pr_std = pred_all.mean().item(), pred_all.std().item()
    
    print(f"GT min: {gt_min:.4f}, max: {gt_max:.4f}, mean: {gt_mean:.4f}, std: {gt_std:.4f}")
    print(f"Pred min: {pr_min:.4f}, max: {pr_max:.4f}, mean: {pr_mean:.4f}, std: {pr_std:.4f}")



    dmin = min(pred_all.min().item(), gt_all.min().item())
    dmax = max(pred_all.max().item(), gt_all.max().item())
    for i in range(rgb_tensor.size(1)):
        rgb_view    = rgb_tensor[0, i].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        gt_d        = depth_tensor[0, i].squeeze(0).cpu().numpy()               # (1, H, W) -> (H, W)
        dpred       = dpreds[0, i].squeeze(0).cpu().numpy()             # (1, H, W) -> (H, W)
        dpred_pretrained = dpreds_pretrained[0, i].squeeze(0).cpu().numpy() # (1, H, W) -> (H, W)

        # Plot the images
        plt.figure(figsize=(20, 5))

        # RGB View
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_view)
        plt.title("RGB View")
        plt.axis("off")

        # Ground Truth Depth
        plt.subplot(1, 4, 2)
        plt.imshow(gt_d, cmap='viridis')
        plt.title("Ground Truth Depth")
        plt.colorbar()
        plt.axis("off")

        # Predicted Depth
        plt.subplot(1, 4, 3)
        plt.imshow(dpred, cmap='viridis')
        plt.title("Predicted Depth")
        plt.colorbar()
        plt.axis("off")
        
        plt.subplot(1, 4, 4)
        plt.imshow(dpred_pretrained, cmap='viridis')
        plt.title("UniDepth Predicted Depth")
        plt.colorbar()
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f'output_{i}.png')
                # 
        # # normalize depths to [0,1] for display
        # gt_norm = (gt_d - dmin) / (dmax - dmin + 1e-6)
        # pr_norm = (dpred - dmin) / (dmax - dmin + 1e-6)



if __name__ == '__main__':
    main()