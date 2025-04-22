from dataset import MultiCamDepthDataset
from losses import DepthLoss
from model import UniDepthV2Finetune
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unidepth.utils.camera import Pinhole as unidepth_Pinhole

import einops
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



'''
TODO
'''


def main():
    # Hyperparams
    npz_dir    = '/project2/yehhh/datasets/RLBench/sim-depth'
    batch_size = 10
    lr         = 1e-2
    epochs     = 20
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp        = 'finetune_20_lr-2schedule_loss1050'
    # exp        = 'test'
    log_dir    = f'runs/{exp}'
    vis_dir    = f'{log_dir}/vis'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Dataset & Loader
    dataset = MultiCamDepthDataset(npz_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Models
    model = UniDepthV2Finetune(device)

    # Freeze backbone
    for p in model.pretrained.parameters():
        p.requires_grad = False
    model.pretrained.eval()

    # Optimizer & loss
    optimizer = optim.Adam(model.correction_head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)  

    
    criterion = DepthLoss(l1_weight=1.0,
                      grad_weight=0.5,
                      normal_weight=0)

    writer = SummaryWriter(log_dir=log_dir)

    # Training
    for ep in range(1, epochs+1):
        running_loss, running_l1, running_grad = 0.0, 0.0, 0
        for batch in loader:
            # batch['rgb']   has shape (B, 4, 3, H, W)
            # batch['depth'] has shape (B, 4, 1, H, W)
            rgb_views   = batch['rgb'].to(device)
            depth_views = batch['depth'].to(device)
            intrinsics  = batch['intrinsics'].to(device)
            dpreds = model(rgb_views, intrinsics)       # (B,4,1,H,W)
            loss, l1, grad, _ = criterion(dpreds, depth_views)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb_views.size(0)
            running_l1   += l1.item() * rgb_views.size(0)
            running_grad += grad.item() * rgb_views.size(0)
        
        scheduler.step()

        avg_loss = running_loss / len(dataset)
        avg_l1   = running_l1 / len(dataset)
        avg_grad = running_grad / len(dataset)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep}/{epochs} — Loss: {avg_loss:.4f} — L1: {avg_l1:.4f} — Grad: {avg_grad:.4f} — Learning Rate: {current_lr}")
        writer.add_scalar('Loss/train', avg_loss, ep)

        with torch.no_grad():
        
            
            gt_all   = depth_views[0]  # (5*1*H*W,)
            pred_all = dpreds[0]

            gt_min, gt_max = gt_all.min().item(), gt_all.max().item()
            gt_mean, gt_std = gt_all.mean().item(), gt_all.std().item()

            pr_min, pr_max = pred_all.min().item(), pred_all.max().item()
            pr_mean, pr_std = pred_all.mean().item(), pred_all.std().item()
            # Mean absolute error
            mae = torch.abs(pred_all - gt_all).mean().item()

            # Log scalars
            writer.add_scalar('Stats/GT_min',  gt_min, ep)
            writer.add_scalar('Stats/GT_max',  gt_max, ep)
            writer.add_scalar('Stats/GT_mean', gt_mean, ep)
            writer.add_scalar('Stats/GT_std',  gt_std, ep)

            writer.add_scalar('Stats/PR_min',  pr_min, ep)
            writer.add_scalar('Stats/PR_max',  pr_max, ep)
            writer.add_scalar('Stats/PR_mean', pr_mean, ep)
            writer.add_scalar('Stats/PR_std',  pr_std, ep)

            dmin = min(pred_all.min().item(), gt_all.min().item())
            dmax = max(pred_all.max().item(), gt_all.max().item())

            dpreds_pretrained = torch.zeros(dpreds.shape, device=device)
            for b in range(dpreds.size(0)):
                for v in range(dpreds.size(1)):
                    # single view rgb: (1,3,H,W)
                    rgb_bv = rgb_views[b, v:v+1] * 255  

                    K_bv = intrinsics[b, v]                # (3,3)
                    camera = unidepth_Pinhole(K=K_bv.unsqueeze(0))

                    out = model.pretrained.infer(rgb_bv, camera=camera)
                    d_bv = out['depth'].unsqueeze(1)           # (1,1,H,W)

                    dpreds_pretrained[b, v] = d_bv

            for i in range(rgb_views.size(1)):
                rgb_view    = rgb_views[0, i].permute(1, 2, 0).cpu().numpy()
                gt_d        = depth_views[0, i].squeeze(0).cpu().numpy() 
                dpred       = dpreds[0, i].squeeze(0).cpu().numpy() 
                dpred_pretrained = dpreds_pretrained[0, i].squeeze(0).cpu().numpy() # (1, H, W) -> (H, W)

                # Plot the images
                fig = plt.figure(figsize=(20, 5))

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
                plt.savefig(os.path.join(vis_dir, f'vis_{ep}_{i}.png'), dpi=300)
                plt.close(fig)

                # normalize depths to [0,1] for display
                gt_norm = (gt_d - dmin) / (dmax - dmin + 1e-6)
                pr_norm = (dpred - dmin) / (dmax - dmin + 1e-6)

                # writer.add_image(f'Vis/RGB_{i}', rgb_view, ep)

                # fig.canvas.draw()
                # width, height = fig.canvas.get_width_height()
                # img = np.frombuffer(fig.canvas.canvas.buffer_rgba(), dtype=np.uint8)
                # img = img.reshape((height, width, 4))  # H x W x 4
                # plt.close(fig)
                # writer.add_image(f'Vis/Depth_{i}', img[:, :, :3], ep, dataformats='HWC')

        if ep % 1 == 0:
            # Save correction head
            torch.save(model.correction_head.state_dict(), os.path.join(log_dir, f'correction_model_{ep}.pth'))
            print("Training complete. Saved correction_net.pth")
    writer.close()



if __name__ == '__main__':
    main()