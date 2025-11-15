import math
import torch
import torch.nn as nn


def make_grid_zyx(Z, Y, X, device=None, dtype=torch.float32):
    """
    Create a regular 3D grid in normalized coordinates [-1, 1]^3
    """
    zz = torch.linspace(-1, 1, Z, device=device, dtype=dtype)
    yy = torch.linspace(-1, 1, Y, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, X, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing='ij'), dim=-1)  # (Z,Y,X,3)
    return grid.view(-1, 3)  # (N,3)


class FourierPosEnc(nn.Module):
    """
    Gaussian-style positional encoding.
    Input:  (N, 3) coords in [-1, 1]
    Output: (N, 9) = [coords, cos(Bx), sin(Bx)]
    """

    def __init__(self):
        super().__init__()
        B = torch.tensor(
            [
                [-8.9113, 19.1638, -8.8575],
                [ 3.9577, 10.2506,  3.8891],
                [12.1395,  5.0181, 12.4339]
            ],
            dtype=torch.float32
        )
        self.register_buffer("B", B, persistent=False)

    def forward(self, C):
        proj = (2.0 * math.pi) * (C @ self.B.T)  # (N, 3)
        return torch.cat([C, torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 9)