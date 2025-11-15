import torch
import torch.nn.functional as F


def rotate_z(coords, theta_rad):
    """
    Rotate 3D coords around the z-axis.
    """
    c, s = torch.cos(theta_rad), torch.sin(theta_rad)
    R = coords.new_tensor(
        [
            [ c, -s, 0.],
            [ s,  c, 0.],
            [0., 0., 1.],
        ]
    )
    return coords @ R.T  # (N, 3)


def coords_to_uv_for_angle(coords, theta_rad):
    """
    Map 3D voxel coords (z, y, x) to 2D (u, v) image coordinates for a given angle.
    """
    if not torch.is_tensor(theta_rad):
        theta_rad = coords.new_tensor(theta_rad)

    # Reorder (z, y, x) -> (x, y, z)
    xyz = torch.stack(
        [coords[:, 2],  # x
         coords[:, 1],  # y
         coords[:, 0]], # z
        dim=-1
    )  # (N, 3)

    # Rotate in x-y plane (inverse transform: volume -> view)
    xyz_rot = rotate_z(xyz, -theta_rad)

    u, v = xyz_rot[:, 0], xyz_rot[:, 1]  # x', y'
    return torch.stack([u, v], dim=-1)   # (N, 2)


def sample_features_single_view(Wi, coords, theta):
    """
    Sample 2D features from a single view at given 3D coordinates.
    """
    B, C, H, W = Wi.shape

    # Case: per-batch angle
    if torch.is_tensor(theta) and theta.ndim == 1 and theta.shape[0] == B:
        feats = []
        for b in range(B):
            uv = coords_to_uv_for_angle(coords, theta[b])          # (N, 2)
            grid = uv.view(1, -1, 1, 2).to(Wi.device).to(Wi.dtype) # (1, N, 1, 2)
            feat = F.grid_sample(
                Wi[b:b+1], grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )                                                       # (1, C, N, 1)
            feat = feat.squeeze(-1).permute(0, 2, 1)               # (1, N, C)
            feats.append(feat)
        return torch.cat(feats, dim=0)                             # (B, N, C)

    # Case: scalar angle, same for all batch elements
    else:
        uv = coords_to_uv_for_angle(coords, theta)                 # (N, 2)
        grid = (
            uv.view(1, -1, 1, 2)
              .expand(B, -1, -1, -1)
              .to(Wi.device)
              .to(Wi.dtype)
        )
        feat = F.grid_sample(
            Wi, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )                                                           # (B, C, N, 1)
        return feat.squeeze(-1).permute(0, 2, 1)                   # (B, N, C)



def sample_features_multi_view_corr(W, angles, coords):

    B, V, C, H, W_ = W.shape
    all_feats = []

    for v in range(V):
        Wi = W[:, v]                 # (B, C, H, W)
        theta = angles[:, v]         # (B,)
        Feat_v = sample_features_single_view(Wi, coords, theta)  # (B, N, C)
        all_feats.append(Feat_v.unsqueeze(1))                    # (B, 1, N, C)

    return torch.cat(all_feats, dim=1)  # (B, V, N, C)