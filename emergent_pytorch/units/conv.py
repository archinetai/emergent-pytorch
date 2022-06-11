import math 
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from ..emergent import Unit

class ConvHead(Unit):

    def __init__(self, image_size: int, out_tokens: int, features: int, activation: nn.Module, channels = 3, lr=0.0005):
        super().__init__(lr = lr) 
        patch_pixels = (image_size ** 2) // out_tokens 
        assert features % patch_pixels == 0, 'Features must be a multiple of: image_size**2 // out_tokens'
        out_channels = features // patch_pixels 
        
        self.channels = channels
        self.patch_size = int(math.sqrt(patch_pixels))
        self.image_size_in_patches = image_size // self.patch_size 
        self.activation = activation 

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, 
                out_channels=channels, 
                kernel_size=3,
                padding=1,
            ),
            activation,
            nn.Conv2d(
                in_channels=channels, 
                out_channels=out_channels, 
                kernel_size=3,
                padding=1,
            )
        )

        self.f_inv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3,
                padding=1,
            ),
            activation,
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=channels, 
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        x = self.activation(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        return x 

    def inverse(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = self.patch_size, p2 = self.patch_size, c = self.channels, h = self.image_size_in_patches, w = self.image_size_in_patches)
        x = self.f_inv(x)
        return x 