import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole as unidepth_Pinhole

class CrossViewSpatioAttentionHead(nn.Module):
    def __init__(self, in_ch=4, d_model=64, nhead=4, num_layers=1, levels=4):
        super().__init__()
        self.levels = levels
        # 1) build a stack of `levels` conv↓2 blocks
        downs = []
        c_in = in_ch
        for i in range(levels):
            downs.append(nn.Conv2d(c_in, d_model,
                                   kernel_size=3, stride=2, padding=1))
            c_in = d_model
        self.downs = nn.Sequential(*downs)

        cv_layer   = nn.TransformerEncoderLayer(d_model, nhead)
        self.cross_view = nn.TransformerEncoder(cv_layer, num_layers)
        sp_layer   = nn.TransformerEncoderLayer(d_model, nhead)
        self.spatial    = nn.TransformerEncoder(sp_layer, num_layers)
        # self.to_delta   = nn.Conv2d(d_model, 1, kernel_size=1)
        # 3) build a stack of `levels` ConvTranspose2d↑2 blocks
        ups = []
        for i in range(levels):
            ups.append(nn.ConvTranspose2d(d_model, d_model,
                                         kernel_size=4, stride=2, padding=1))
        self.ups = nn.Sequential(*ups)

        # 4) final 1×1 to get residual
        self.final = nn.Conv2d(d_model, 1, kernel_size=1)

        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B*V, in_ch, 1, 1)
            nn.Flatten(start_dim=1),   # (B*V, in_ch)
            nn.Linear(in_ch, 1),       # (B*V, 1)
            nn.Softplus()              # ensure scale>0
        )

    def forward(self, x):
        """
        x: (B, V, C=4, H, W)
        returns delta: (B, V, 1, H, W)
        """
        B, V, C, H, W = x.shape

        x_flat = einops.rearrange(x, 'b v c h w -> (b v) c h w')
        scale = self.scale_head(x_flat)        # → (B*V, 1)
        scale = einops.rearrange(scale, '(b v) 1 -> b v 1 1 1', b=B, v=V)

        # 1) lift + downsample → (B*V, d_model, H2, W2)
        feat = einops.rearrange(x, 'b v c h w -> (b v) c h w')
        feat = self.downs(feat)
        H2, W2 = feat.shape[2], feat.shape[3]

        # 2) cross‑view at each pixel
        #    (B*V, d, H2, W2) -> (B, H2, W2, V, d)
        f_cv = einops.rearrange(feat, '(b v) d h2 w2 -> b h2 w2 v d', b=B, v=V)
        #    -> (V, B*H2*W2, d)
        tokens_cv = einops.rearrange(f_cv, 'b h2 w2 v d -> v (b h2 w2) d')
        
        att_cv    = self.cross_view(tokens_cv)  # (V, B*H2*W2, d)
        #    -> (B, V, d, H2, W2)
        f_cv2 = einops.rearrange(att_cv, 'v (b h2 w2) d -> b v d h2 w2',
                                 b=B, h2=H2, w2=W2)

        # 3) spatial within each view
        #    -> (H2*W2, B*V, d)
        tokens_sp = einops.rearrange(f_cv2, 'b v d h2 w2 -> (h2 w2) (b v) d')
        att_sp    = self.spatial(tokens_sp)     # (H2*W2, B*V, d)
        #    -> (B, V, d, H2, W2)
        f_sp2 = einops.rearrange(att_sp, '(h2 w2) (b v) d -> b v d h2 w2',
                                 b=B, v=V, h2=H2, w2=W2)

        # 4) upsample via ConvTranspose2d
        #    (B, V, d, H2, W2) -> (B*V, d, H2, W2)
        f_up = einops.rearrange(f_sp2, 'b v d h2 w2 -> (b v) d h2 w2')
        f_up = self.ups(f_up)

        delta_flat = self.final(f_up)                                 # (B*V,1,H,W)
        delta = einops.rearrange(delta_flat, '(b v) 1 h w -> b v 1 h w',
                                 b=B, v=V)

        return scale, delta


class UniDepthV2Finetune(nn.Module):
    def __init__(self, device, correction_head_weights=None):
        super().__init__()
        self.device = device

        # load & freeze the pretrained backbone
        self.pretrained = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vitl14"
        ).to(device)
        self.pretrained.eval()
        for p in self.pretrained.parameters():
            p.requires_grad = False

        self.correction_head = CrossViewSpatioAttentionHead().to(device)

        if correction_head_weights is not None:
            # load weights into correction head
            state_dict = torch.load(correction_head_weights, map_location=device)
            self.correction_head.load_state_dict(state_dict)

    def forward(self, rgb, intrinsics=None):
        """
        Args:
          rgb:        (B, V=4, 3, H, W)
          intrinsics: (B, V=4, 3, 3) or None
        Returns:
          depth_corrected: (B, V=4, 1, H, W)
        """
        B, V, C, H, W = rgb.shape
        assert V == 4 and C == 3


        device = rgb.device

        # allocate tensor for predicted depths
        dpred = torch.zeros(B, V, 1, H, W, device=device)

        # loop over batch and view
        for b in range(B):
            for v in range(V):
                # single view rgb: (1,3,H,W)
                rgb_bv = rgb[b, v:v+1] * 255  

                # build camera if intrinsics given
                if intrinsics is not None:
                    K_bv = intrinsics[b, v]                # (3,3)
                    camera = unidepth_Pinhole(K=K_bv.unsqueeze(0))
                else:
                    camera = None

                # infer single depth
                with torch.no_grad():
                    out = self.pretrained.infer(rgb_bv, camera=camera)
                d_bv = out['depth'].unsqueeze(1)           # (1,1,H,W)

                dpred[b, v] = d_bv
        x = torch.cat([rgb, dpred], dim=2)              # (B, V=4, 4, H, W)
        scale, delta = self.correction_head(x)                 # (B, V=4, 1, H, W)

        return scale * dpred + delta                            # (B, V=4, 1, H, W)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniDepthV2Finetune(device)

    # dummy batch: B=2, V=4 views, 3-ch RGB, 32×32
    B = 10
    rgb = torch.ones(B,4,3,256,256, device=device)
    K   = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(B,4,1,1)

    out = model(rgb, intrinsics=K)
    print('output shape:', out.shape)  # should be (2, 5, 1, 32, 32)