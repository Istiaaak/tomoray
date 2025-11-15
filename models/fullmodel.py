import torch
import torch.nn as nn

from .pos_encoder import FourierPosEnc, make_grid_zyx
from .sampling import sample_features_single_view
from .inr import INR_ResNet


class MultiViewCT(nn.Module):
    """
    Multi-view CT reconstruction model:
      - 2D U-Net backbone on each DRR (per view),
      - feature sampling back into 3D at arbitrary coordinates,
      - per-view INR f_rho,
      - global INR h_tau aggregating per-view embeddings.
    """

    def __init__(
        self,
        unet_2d,
        Z=128, Y=128, X=128,
        feat_ch=128,
        pe_dim=9,
        inr_hidden=128,
        inr_blocks_view=3,    # number of SIREN blocks in f_rho
        inr_blocks_global=2,  # number of SIREN blocks in h_tau
        emb_dim=128,          # embedding dimension per view r_i(p)
        inr_out=1,
    ):
        super().__init__()
        # 2D backbone: (B*V, 1, H, W) -> (B*V, C, H, W)
        self.backbone = unet_2d
        # Positional encoding in 3D
        self.pe       = FourierPosEnc()

        self.Z, self.Y, self.X = Z, Y, X
        self.feat_ch  = feat_ch
        self.pe_dim   = pe_dim
        self.emb_dim  = emb_dim

        # f_rho: per-view MLP, maps [feat_i(p), PE(p)] -> r_i(p)
        in_dim_view = feat_ch + pe_dim
        self.inr_view = INR_ResNet(
            in_dim=in_dim_view,
            hidden=inr_hidden,
            num_blocks=inr_blocks_view,
            out_dim=emb_dim,
        )

        # h_tau: global MLP, maps r_bar(p) -> CT intensity
        in_dim_global = emb_dim
        self.inr_global = INR_ResNet(
            in_dim=in_dim_global,
            hidden=inr_hidden,
            num_blocks=inr_blocks_global,
            out_dim=inr_out,
        )

        # Precompute 3D coordinates (N = Z * Y * X)
        coords = make_grid_zyx(Z, Y, X, device=torch.device("cpu"))  # (N, 3)
        self.register_buffer("coords", coords, persistent=False)


    def forward_points(self, views, angles, idx, chunk=4096):
        """
        Predict CT values at a subset of voxels.

        Args:
            views:  (B, V, 1, H, W) DRR images.
            angles: (B, V) view angles in radians.
            idx:    (n_points,) indices in [0, Z*Y*X), consistent with self.coords
                    and with ct.view(B, -1).
            chunk:  number of points processed per chunk (for memory).

        Returns:
            (B, n_points) predicted CT values at selected voxels.
        """
        B, V, _, H, W = views.shape
        device = views.device
        dtype  = views.dtype

        # 1) Run the 2D U-Net on all views
        x = views.view(B * V, 1, H, W)          # (B*V, 1, H, W)
        feats = self.backbone(x)                # (B*V, C, H, W)
        C = feats.shape[1]
        assert C == self.feat_ch, f"feat_ch mismatch, got {C}, expected {self.feat_ch}"
        feats = feats.view(B, V, C, H, W)       # (B, V, C, H, W)

        # 2) Select coordinates + positional encoding
        coords_all = self.coords.to(device=device)   # (N, 3)
        coords = coords_all[idx]                     # (n_points, 3)
        Np = coords.shape[0]
        pe = self.pe(coords).to(device=device, dtype=dtype)  # (n_points, pe_dim)

        out_chunks = []

        # 3) Process points in chunks for memory efficiency
        for start in range(0, Np, chunk):
            end = min(start + chunk, Np)
            n   = end - start

            coords_c = coords[start:end]        # (n, 3)
            pe_c     = pe[start:end]            # (n, pe_dim)

            # Accumulate embeddings r_i(p) over views
            r_sum = torch.zeros(B, n, self.emb_dim, device=device, dtype=dtype)

            for v in range(V):
                Wi    = feats[:, v]             # (B, C, H, W)
                theta = angles[:, v]            # (B,)

                # 2D feature sampling: (B, n, C) at these voxels
                f_c = sample_features_single_view(Wi, coords_c, theta)  # (B, n, C)

                # Concatenate [feat_i(p), PE(p)]
                pe_exp = pe_c.unsqueeze(0).expand(B, -1, -1)            # (B, n, pe_dim)
                x_c    = torch.cat([f_c, pe_exp], dim=-1)               # (B, n, C + pe_dim)

                x_flat = x_c.reshape(B * n, -1)                         # (B*n, D_in)
                r_flat = self.inr_view(x_flat)                          # (B*n, emb_dim)
                r_i    = r_flat.view(B, n, self.emb_dim)                # (B, n, emb_dim)

                r_sum += r_i

            # Average embeddings over views: r_bar(p)
            r_mean = r_sum / float(V)                                   # (B, n, emb_dim)

            # Global MLP h_tau(r_bar(p))
            y_flat = self.inr_global(
                r_mean.reshape(B * n, self.emb_dim)
            )                                                           # (B*n, 1)
            y_c    = y_flat.view(B, n)                                  # (B, n)
            out_chunks.append(y_c)

        # Concatenate all chunks → (B, n_points)
        y = torch.cat(out_chunks, dim=1)                                # (B, n_points)
        return y


    def forward(self, views, angles, chunk=4096):
        """
        Reconstruct the full CT volume (B, 1, Z, Y, X).

        This is more expensive in memory/time and is usually used with
        `torch.no_grad()` for validation / visualization.

        Args:
            views:  (B, V, 1, H, W)
            angles: (B, V)
            chunk:  number of points per chunk in forward_points.

        Returns:
            (B, 1, Z, Y, X) reconstructed volume.
        """
        B, V, _, H, W = views.shape
        device = views.device

        N_vox = self.coords.shape[0]   # Z*Y*X
        idx_all = torch.arange(N_vox, device=device, dtype=torch.long)

        # Predict all voxels: (B, N_vox)
        y_flat = self.forward_points(views, angles, idx_all, chunk=chunk)  # (B, N_vox)

        # Reshape into 3D volume
        y = y_flat.view(B, 1, self.Z, self.Y, self.X)                      # (B, 1, Z, Y, X)
        return y