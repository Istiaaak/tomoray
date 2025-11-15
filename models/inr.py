import math
import torch
import torch.nn as nn


class SirenLinear(nn.Module):
    """
    Linear layer with SIREN-style initialization and scaling.
    """

    def __init__(self, in_feat, out_feat, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.is_first = is_first
        self.omega_0 = omega_0
        self._init(in_feat)

    def _init(self, in_feat):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_feat, 1 / in_feat)
            else:
                bound = math.sqrt(6 / in_feat) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        # Output is scaled by omega_0, as in SIREN
        return self.omega_0 * self.linear(x)


class ResBlockSiren(nn.Module):
    """
    Simple residual block using SIREN-style layers and sine activation.
    """

    def __init__(self, dim):
        super().__init__()
        self.fc = SirenLinear(dim, dim, is_first=False, omega_0=50)

    def forward(self, x):
        dx = torch.sin(self.fc(x))
        dx = self.fc(dx)
        return torch.sin(x + dx)


class INR_ResNet(nn.Module):
    """
    ResNet-style implicit network with SIREN layers.
    """

    def __init__(self, in_dim, hidden=128, num_blocks=3, out_dim=1):
        super().__init__()
        self.fc_in = SirenLinear(in_dim, hidden, is_first=True, omega_0=50)
        self.blocks = nn.ModuleList(
            [ResBlockSiren(hidden) for _ in range(num_blocks)]
        )
        self.fc_mid = nn.Sequential(
            *[
                SirenLinear(hidden, hidden, is_first=False, omega_0=50)
                for _ in range(2)
            ]
        )
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):  # x: (B*N, in_dim)
        x = torch.sin(self.fc_in(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_mid(x)
        return self.fc_out(x)