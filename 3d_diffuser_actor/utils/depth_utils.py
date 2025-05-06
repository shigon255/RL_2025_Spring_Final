import einops
import glob
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# import depth_pro
# from unik3d.models import UniK3D
from unidepth.models import UniDepthV1, UniDepthV2
# from unik3d.utils.camera import Pinhole as unik3d_Pinhole
from unidepth.utils.camera import Pinhole as unidepth_Pinhole


## YCH: import finetune depth model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UniDepthFinetune.model import UniDepthV2Finetune


def init_depth_pro(device='cuda'):
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    return model, transform

def init_unidepth(device='cuda', name='unidepth-v2-vitl14'):
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    model = model.to(device)
    model.eval()
    return model, None

def init_unik3d(device='cuda', name='unik3d-vitl'):
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model = model.to(device)
    model.eval()
    return model, None

## YCH: init unidepth finetune model
def init_unidepthfinetune(device='cuda', name='unidepth-v2-vitl14', \
                            finetune_weights='./UniDepthFinetune/runs/correction_model.pth'):
    finetune_weights = "./UniDepthFinetune/runs/finetune_lr-3schedule_loss110_weightedl1_4camera/correction_model_6.pth"
    model = UniDepthV2Finetune(device, correction_head_weights=finetune_weights)
    model = model.to(device)
    model.eval()
    return model, None

## YCH: init unidepth full finetune model
def init_unidepthfullfinetune(device='cuda', name='unidepth-v2-vitl14', \
                            finetune_weights='/data1/yehhh_/RL_2025_Spring_Final/UniDepth/tmp/wandb/run-20250505_033753-27kv0i1l/files/pytorch_model_1000.bin'):
    # 0.0355
    # finetune_weights = "/data1/yehhh_/RL_2025_Spring_Final/UniDepth/tmp/wandb/run-20250505_094342-gv3e99bt/files/pytorch_model_250.bin"
    finetune_weights = "/data1/yehhh_/RL_2025_Spring_Final/UniDepth/tmp/wandb/run-20250506_051256-front_1000/files/pytorch_model_400.bin"
    # 0.0332 
    with open("/data1/yehhh_/RL_2025_Spring_Final/UniDepth/configs/config_v2_vitl14.json") as f:
        config = json.load(f)
    
    model = UniDepthV2(config)
    info = model.load_state_dict(torch.load(finetune_weights), strict=False)
    print(f"UniDepth_v2_vitl14 is loaded with:")
    print(f"\t missing keys: {info.missing_keys}")
    print(f"\t additional keys: {info.unexpected_keys}")
    model = model.to(device)
    model.eval()
    return model, None

def predict_depth_pro(depth_model,
                  depth_model_transform,
                  rgb_img,
                  f_px = None,
                  intrinsics = None, # dummy
) -> np.ndarray:
    """
    Predict depth using DepthPro given an RGB image (H, W, 3) and a optional focal length in pixels.
    """
    if f_px is not None:
        f_px = torch.Tensor([f_px]).to(depth_model.device)
    
    # Load and preprocess an image.
    rgb_img = depth_model_transform(rgb_img)

    # Run inference.
    prediction = depth_model.infer(rgb_img, f_px=f_px)
    depth = prediction["depth"].cpu().numpy()  # Depth in [m].
    # focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    return depth

def predict_depth_unidepth(depth_model,
                           rgb_img,
                           intrinsics = None,
                           f_px = None, # dummy
                           depth_model_transform=None, # dummy
) -> np.ndarray:
    """
    Predict depth using UniDepth given an RGB image (H, W, 3) and a optional focal length in pixels.
    depth_model_transform and f_px should be None for this model, so just ignore it.
    
    intrinsic is optional, and it should be a 3x3 numpy array.
    """
    camera = None
    if intrinsics is not None:
        intrinsics = torch.from_numpy(intrinsics).to(depth_model.device)
        camera = unidepth_Pinhole(K=intrinsics)
    
    rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1) # C, H, W
    predictions = depth_model.infer(rgb_img, camera=camera)
    depth = predictions["depth"]
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()
    
    return depth

## YCH: predict depth for unidepth finetune model
def predict_depth_unidepthfinetune(depth_model,
                           rgb_imgs = None,
                           intrinsics = None,
                           f_px = None, # dummy
                           depth_model_transform=None, # dummy
) -> np.ndarray:
    """
    Args:
        rgb_imgs: (B, V, H, W, 3), V is the number of views, in range [0, 255]
        intrinsics: (B, V, 3, 3)
    Returns:
        depth: (B, V, H, W)
    """

    rgb_imgs = (torch.from_numpy((rgb_imgs[None]).astype(np.float32)) / 255.0).to(depth_model.device) 
    rgb_imgs = einops.rearrange(rgb_imgs, 'B V H W C -> B V C H W')
    intrinsics = torch.from_numpy((intrinsics[None]).astype(np.float32)).to(depth_model.device)
    with torch.no_grad():
        dpreds = depth_model(rgb_imgs, intrinsics)       # (B,4,1,H,W)
    
    dpreds = dpreds.squeeze(2).cpu().numpy()  # (B, V, H, W)
    
    return dpreds[0]

## YCH: predict depth for unidepth finetune model
def predict_depth_unidepthfullfinetune(depth_model,
                           rgb_imgs = None,
                           intrinsics = None,
                           f_px = None, # dummy
                           depth_model_transform=None, # dummy
) -> np.ndarray:
    """
    Args:
        rgb_imgs: (B, H, W, 3), V is the number of views, in range [0, 255]
        intrinsics: (B, 3, 3)
    Returns:
        depth: (B, H, W)
    """
    
    rgb_imgs = (torch.from_numpy(rgb_imgs.astype(np.float32))).to(depth_model.device) 
    rgb_imgs = einops.rearrange(rgb_imgs, 'B H W C -> B C H W')
    
    depths = np.zeros((rgb_imgs.shape[0], rgb_imgs.shape[2], rgb_imgs.shape[3]), dtype=np.float32)
    for i, (rgb_img, intrinsic) in enumerate(zip(rgb_imgs, intrinsics)):
        intrinsic = torch.from_numpy(intrinsic).to(depth_model.device)
        camera = unidepth_Pinhole(K=intrinsic)
        predictions = depth_model.infer(rgb_img, camera=camera)
        depth = predictions["depth"]
        depths[i] = depth.squeeze(0).squeeze(0).cpu().numpy()
    
    return depths

def predict_depth_unik3d(depth_model,
                        rgb_img,
                        intrinsics = None,
                        f_px = None, # dummy
                        depth_model_transform=None, # dummy
    ) -> np.ndarray:
    """
    Predict depth using UniK3D given an RGB image (H, W, 3) and a optional focal length in pixels.
    depth_model_transform and f_px should be None for this model, so just ignore it.
    
    intrinsic is optional, and it should be a 3x3 numpy array.
    """
    camera = None
    if intrinsics is not None:
        intrinsics = torch.from_numpy(intrinsics).to(depth_model.device)
        camera = unik3d_Pinhole(K=intrinsics)
    
    rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1) # C, H, W
    predictions = depth_model.infer(rgb_img, camera=camera)
    depth = predictions["depth"]
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()
    
    return depth

depth_init_functions = {
    'depth_pro': init_depth_pro,
    'unidepth': init_unidepth,
    'unik3d': init_unik3d,
    'unidepthfinetune': init_unidepthfinetune,
    'unidepthfullfinetune': init_unidepthfullfinetune,
}

depth_predict_functions = {
    'depth_pro': predict_depth_pro,
    'unidepth': predict_depth_unidepth,
    'unik3d': predict_depth_unik3d,
    'unidepthfinetune': predict_depth_unidepthfinetune,
    'unidepthfullfinetune': predict_depth_unidepthfullfinetune,
}



if __name__ == '__main__':

    cams = ['left_shoulder', 'right_shoulder', 'wrist', 'front']
    intrinsics_cam = np.array([
        [[351.6771208,   0.0,         128.0],
        [  0.0,       351.6771208,   128.0],
        [  0.0,         0.0,           1.0]],
        [[351.6771208,   0.0,         128.0],
        [  0.0,       351.6771208,   128.0],
        [  0.0,         0.0,           1.0]],
        [[221.70249591,  0.0,         128.0],
        [  0.0,       221.70249591,  128.0],
        [  0.0,         0.0,           1.0]],
        [[351.6771208,   0.0,         128.0],
        [  0.0,       351.6771208,   128.0],
        [  0.0,         0.0,           1.0]],
    ], dtype=np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("== Initializing model ==")
    model, _ = init_unidepthfullfinetune(device=device)

    npz_dir = '/project2/yehhh/datasets/RLBench/sim-depth_m'
    # grab a few .npz files
    all_npzs = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
    sel = all_npzs
    N = len(sel)
    # N = 10
    B = len(cams)

    
    # print(f"\n== Loading {B} samples from NPZ ==")
    # placeholder arrays
    example = np.load(sel[0])
    H, W = example[cams[0] + '_rgb'].shape[:2]

    rgb_batch = np.zeros((B, H, W, 3), dtype=np.uint8)
    gt_depth  = np.zeros((B, H, W), dtype=np.float32)
    # K_batch   = np.tile(intrinsics_cam, (B, 1, 1))
    K_batch   = intrinsics_cam
    
    abs_err = np.zeros((4,), dtype=np.float32)
    for i, path in enumerate(sel[:N]):
        data = np.load(path)
        # print(f"  {os.path.basename(path)} → size {(H, W)}")
        for v, cam in enumerate(cams):
            rgb_batch[v]  = data[f'{cam}_rgb']
            # depth_np in meters → keep as float32
            gt_depth[v]   = data[f'{cam}_depth']
            # intrinsics already filled

        # print("\n== Running prediction ==")
        preds = predict_depth_unidepthfullfinetune(model, rgb_batch, K_batch)
        abs_err += np.mean(np.abs(preds - gt_depth), axis=(1, 2))
        if i == N - 1:
            for idx in range(min(4, rgb_batch.shape[0])):
                # raw RGB is uint8 H×W×3
                rgb = rgb_batch[idx]  

                # ground truth & prediction are float arrays H×W
                depth_gt   = gt_depth[idx]
                depth_pred = preds[idx]

                plt.figure(figsize=(15, 5))

                # RGB
                plt.subplot(1, 3, 1)
                plt.imshow(rgb)
                plt.title("RGB View")
                plt.axis("off")

                # GT depth
                plt.subplot(1, 3, 2)
                im2 = plt.imshow(depth_gt, cmap="viridis")
                plt.title("GT Depth")
                plt.axis("off")
                plt.colorbar(im2, fraction=0.046, pad=0.04)

                # Predicted depth
                plt.subplot(1, 3, 3)
                im3 = plt.imshow(depth_pred, cmap="viridis")
                plt.title("Predicted Depth")
                plt.axis("off")
                plt.colorbar(im3, fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(f'./vis_{idx:03d}.png')
                plt.close()
    
    for i in range(abs_err.shape[0]):
        print(f"{i}th camera error: {abs_err[i] / N:.4f}")
    print(f"Total abs error: {(np.sum(abs_err) / N / B):.4f}")

    # # shape checks
    # assert preds.shape == gt_depth.shape, (
    #     f"Pred shape {preds.shape} != GT shape {gt_depth.shape}"
    # )
    # print(f"\nOutput shape OK: {preds.shape}")

    # # finite check
    # assert np.all(np.isfinite(preds)), "Found non-finite values in output!"
    # print("No NaNs/Infs in predictions.")

    # # simple error metrics
    # abs_err = np.abs(preds - gt_depth)
    # print(f"Error stats: min={abs_err.min():.4f}, "
    #       f"max={abs_err.max():.4f}, mean={abs_err.mean():.4f}")

    # print("\n✅ Full NPZ→model round-trip sanity check passed!")