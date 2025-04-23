import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthLoss(nn.Module):
    """
    Combined depth loss = L1 + gradient loss + normal loss,
    accepting multi‑view inputs of shape (B, V, 1, H, W).
    """
    def __init__(self, l1_weight=1.0, grad_weight=1.0, normal_weight=1.0):
        super().__init__()
        self.l1_weight     = l1_weight
        self.grad_weight   = grad_weight
        self.normal_weight = normal_weight
        self.eps           = 1e-6

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            pred: (B, V, 1, H, W) — predicted depth
            gt:   (B, V, 1, H, W) — ground‑truth depth
        Returns:
            Scalar loss
        """
        B, V, C, H, W = pred.shape
        assert C == 1

        # flatten batch & view -> (N,1,H,W), where N = B*V
        N = B * V
        pred = einops.rearrange(pred, 'B V 1 H W -> (B V) 1 H W')
        gt   = einops.rearrange(gt, 'B V 1 H W -> (B V) 1 H W')

        # 1) L1 loss
        # Normalize gt into [0,1]
        gt_min = gt.amin(dim=[1,2,3], keepdim=True)
        gt_max = gt.amax(dim=[1,2,3], keepdim=True)
        gt_norm = (gt - gt_min) / (gt_max - gt_min + self.eps)

        # Choose weight function
        # w = 1.0 / (gt_norm + self.eps)
        w = 1.0 - gt_norm

        # Normalize weights so that mean(w)=1
        w = w / w.mean()

        # Compute weighted L1
        l1 = (w * torch.abs(pred - gt)).mean()
        # l1 = F.l1_loss(pred, gt, reduction='mean')


        # 2) gradient loss
        #  ── horizontal gradients
        pdx = pred[..., :, :, 1:] - pred[..., :, :, :-1]  # (N,1,H,W-1)
        gdx =   gt[..., :, :, 1:] -   gt[..., :, :, :-1]
        #  ── vertical gradients
        pdy = pred[..., :, 1:, :] - pred[..., :, :-1, :]  # (N,1,H-1,W)
        gdy =   gt[..., :, 1:, :] -   gt[..., :, :-1, :]

        grad_loss = (pdx - gdx).abs().mean() + (pdy - gdy).abs().mean()

        normal_loss = torch.zeros(1, device=pred.device)
        if self.normal_weight > 0:
            # 3) normal loss (on interior pixels)
            # crop to valid region: size (N,1,H-1,W-1)
            dx = pdx[..., 0:H-1, 0:W-1]
            dy = pdy[..., 0:H-1, 0:W-1]
            dz = torch.ones_like(dx)

            n_pred = torch.cat([-dx, -dy, dz], dim=1)  # (N,3,H-1,W-1)
            n_pred = n_pred / (n_pred.norm(dim=1, keepdim=True) + self.eps)

            dx_gt = gdx[..., 0:H-1, 0:W-1]
            dy_gt = gdy[..., 0:H-1, 0:W-1]
            dz_gt = torch.ones_like(dx_gt)

            n_gt = torch.cat([-dx_gt, -dy_gt, dz_gt], dim=1)
            n_gt = n_gt / (n_gt.norm(dim=1, keepdim=True) + self.eps)

            cos = (n_pred * n_gt).sum(dim=1)  # (N,H-1,W-1)
            normal_loss = (1 - cos).mean()

        # weighted sum
        loss = ( self.l1_weight     * l1
               + self.grad_weight   * grad_loss
               + self.normal_weight * normal_loss )
        
        print(f"Loss: total={loss.item():.4f}, l1={l1.item():.4f}, grad={grad_loss.item():.4f}, normal={normal_loss.item():.4f}")
        if torch.isnan(loss):
            print("Loss is NaN!")
            print(f"Loss: total={loss.item():.4f}, l1={l1.item():.4f}, grad={grad_loss.item():.4f}, normal={normal_loss.item():.4f}")

        return loss, self.l1_weight * l1, self.grad_weight * grad_loss, self.normal_weight * normal_loss

    

if __name__ == "__main__":
    B, V, H, W = 2, 4, 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create random GT
    gt = torch.rand(B, V, 1, H, W, device=device)

    # 1) pred == gt → loss should be zero
    pred = gt.clone()
    criterion = DepthLoss(l1_weight=1.0, grad_weight=1.0, normal_weight=1.0).to(device)
    loss_zero, _, _, _ = criterion(pred, gt)
    print(f"Loss when pred==gt: {loss_zero.item():.6f}")
    assert torch.isclose(loss_zero, torch.tensor(0., device=device), atol=1e-5), "Loss must be zero when pred==gt"

    # 2) pred = gt + c → gradient & normal losses still zero, so loss == c
    c = 0.1234
    pred2 = gt + c
    loss_offset, _, _, _ = criterion(pred2, gt)
    print(f"Loss when pred=gt+{c}: {loss_offset.item():.6f}")
    assert abs(loss_offset - c) < 1e-4, "Loss should equal the constant offset when only L1 is active"

    print("✅ DepthLoss sanity checks passed!")
