from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.si_block import SIBlocks
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel

class SINO(BaseModel, name='SINO'):
    """Spline-Integral Neural Operator. The SINO is useful for solving IVPs.

    Parameters
    ----------


    Other parameters
    ----------------


    Examples
    --------


    References
    ----------

    """

    def __init__(
		  self,
      in_channels: int,
      out_channels: int,
      hidden_channels: int,
      num_knots: int,
      n_dim: int=64,
      n_layers: int=4,
      lifting_channel_ratio: int=2,
      projection_channel_ratio: int=2,
      non_linearity: nn.Module=F.gelu
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_knots = num_knots
        self.n_dim = n_dim
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        # self.sino_blocks = SIBlocks(in_channels=self.in_channels, out_channels=self.out_channels, num_knots=self.num_knots)
        # self.sino_blocks = nn.ModuleList([
        #     SIBlocks(in_channels=self.in_channels, out_channels=self.out_channels, num_knots=self.num_knots)
        #     for _ in range(self.n_layers)
        # ])
        self.sino_blocks = nn.ModuleList([
            SIBlocks(in_channels=self.lifting_channels, out_channels=self.projection_channels, num_knots=self.num_knots)
            for _ in range(self.n_layers)
        ])
            
        # define lifting and projection networks
        lifting_in_channels = self.in_channels
        self.lifting = ChannelMLP(
                in_channels=lifting_in_channels+2,
                out_channels=self.lifting_channels,
                hidden_channels=self.lifting_channels * 2,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        
        self.projection = ChannelMLP(
            in_channels=self.projection_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels * 2,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def get_grid(self, B, H, W, device):
        gridx = torch.linspace(0, 1, H, device=device).reshape(1, 1, H, 1).repeat(B, 1, 1, W)
        gridy = torch.linspace(0, 1, W, device=device).reshape(1, 1, 1, W).repeat(B, 1, H, 1)
        return torch.cat([gridx, gridy], dim=1)  # (B, 2, H, W)

    def forward(self, x, **kwargs):
        """SINO's forward pass

        1. Sends inputs through a lifting layer to a high-dimensional latent space

        2. Applies `n_layers` SINO layers in sequence 

        3. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        
        """
        B, C, H, W = x.shape
        
        grid = self.get_grid(B, H, W, x.device)

        x = torch.cat([x, grid], dim=1)

        x = self.lifting(x)

        # x = x.view(B, self.hidden_channels, H * W).permute(0, 2, 1)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.lifting_channels)

        # for _ in range(n_layers)
        for i, layer in enumerate(self.sino_blocks):
            # x = self.sino_blocks(x)

            # PRINT H_NET weights
            # for j, l in enumerate(layer.h_net):
                # if isinstance(l, nn.Linear):
                    # print(f"Layer {j} weights:\n", l.weight.data)
            layer.plot_and_save_spline(save_path="SPLINE_PLOTS/learned_spline.png")
            x = x + layer(x)


        # x = x.permute(0, 2, 1).view(B, self.hidden_channels, H, W)
        # x = x.view(B, H, W, self.projection_channels).permute(0, 3, 1, 2)

        # # x = self.projection(x)

        # x = self.projection(x.permute(0, 2, 3, 1).reshape(B, H, W, self.projection_channels))  # → (B, N, out_channels)
        # x = x.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)  # → (B, out_channels, H, W)

        x = x.view(B, H, W, self.projection_channels)                # → (B, H, W, C)
        x = x.permute(0, 3, 1, 2).reshape(B, self.projection_channels, H * W)  # → (B, C, N)

        x = self.projection(x)                                       # projection: (B, C_out, N)
        x = x.view(B, self.out_channels, H, W)

        return x