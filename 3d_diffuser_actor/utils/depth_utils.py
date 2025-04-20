import torch
import numpy as np
# import depth_pro
# from unik3d.models import UniK3D
from unidepth.models import UniDepthV1, UniDepthV2
# from unik3d.utils.camera import Pinhole as unik3d_Pinhole
from unidepth.utils.camera import Pinhole as unidepth_Pinhole


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
}

depth_predict_functions = {
    'depth_pro': predict_depth_pro,
    'unidepth': predict_depth_unidepth,
    'unik3d': predict_depth_unik3d,
}