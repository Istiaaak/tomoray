import torch
import torch.nn.functional as F


@torch.no_grad()
def _norm_like(a, b, eps=1e-6):
    """
    Optionally match the intensity range of `a` to `b` (min-max per image).

    This can stabilize training if there are small intensity drifts.
    """
    bmin, bmax = b.amin(dim=(-2, -1), keepdim=True), b.amax(dim=(-2, -1), keepdim=True)
    amin, amax = a.amin(dim=(-2, -1), keepdim=True), a.amax(dim=(-2, -1), keepdim=True)
    a = (a - amin) / (amax - amin + eps)
    a = a * (bmax - bmin) + bmin
    return a


def _coords_rot_uv(theta, H, W, device):
    """
    Build a 2D grid (H, W, 2) in normalized [-1, 1] coords for grid_sample 3D,
    corresponding to a rotation in the (x, y) plane.

    Uses align_corners=True convention.
    """
    v = torch.linspace(-1, 1, steps=H, device=device)   # y
    u = torch.linspace(-1, 1, steps=W, device=device)   # x
    yy, xx = torch.meshgrid(v, u, indexing="ij")        # (H, W)

    c = torch.cos(-theta)  # inverse transform: volume -> view
    s = torch.sin(-theta)

    x =  c * xx + s * yy
    y = -s * xx + c * yy
    return x, y  # (H, W), (H, W)


def render_drr_orthographic(vol_b1dhw, thetas, out_hw=None, steps=None):
    """
    Approximate DRR rendering by line integration using grid_sample in 3D.

    Args:
        vol_b1dhw: (B, 1, D, H, W) reconstructed volume.
        thetas:    (B, V) angles in radians for each view.
        out_hw:    (H_out, W_out) DRR resolution (defaults to input H, W).
        steps:     number of samples along z-axis (defaults to D).

    Returns:
        (B, V, 1, H_out, W_out) DRRs for each batch and view.
    """
    B, _, D, H, W = vol_b1dhw.shape
    Hout = out_hw[0] if out_hw else H
    Wout = out_hw[1] if out_hw else W
    S = steps or D

    zz = torch.linspace(-1, 1, steps=S, device=vol_b1dhw.device)   # (S,)
    drrs = []

    for v in range(thetas.shape[1]):
        th = thetas[:, v]                                          # (B,)
        grids = []

        # Build per-batch 3D grid
        for b in range(B):
            x, y = _coords_rot_uv(th[b], Hout, Wout, vol_b1dhw.device)   # (H, W)
            x_b = x.expand(S, Hout, Wout)
            y_b = y.expand(S, Hout, Wout)
            z_b = zz.view(S, 1, 1).expand(S, Hout, Wout)
            grid = torch.stack([x_b, y_b, z_b], dim=-1).unsqueeze(0)     # (1, S, H, W, 3)
            grids.append(grid)

        grid = torch.cat(grids, dim=0)                                    # (B, S, H, W, 3)

        # Sample volume along z to approximate line integrals
        samp = F.grid_sample(
            vol_b1dhw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                                                  # (B, 1, S, H, W)
        drr = samp.mean(dim=2, keepdim=False)                              # (B, 1, H, W)
        drrs.append(drr.unsqueeze(1))                                      # (B, 1, 1, H, W)

    return torch.cat(drrs, dim=1)                                          # (B, V, 1, H, W)


def projection_loss_from_angles(pred_vol, views, angles, lambda_align=1.0, use_norm=True):
    """
    Projection-domain loss between predicted volume and input DRRs.

    Args:
        pred_vol: (B, 1, D, H, W) predicted CT volume.
        views:    (B, V, 1, H, W) target DRRs.
        angles:   (B, V) angles in radians.
        use_norm: if True, re-scale predicted DRRs to match target ranges.

    Returns:
        scalar MSE loss.
    """
    B, V, _, H, W = views.shape
    pred_drr = render_drr_orthographic(pred_vol, angles, out_hw=(H, W))  # (B, V, 1, H, W)

    if use_norm:
        with torch.no_grad():
            pred_drr = _norm_like(pred_drr, views)

    return F.mse_loss(pred_drr, views)