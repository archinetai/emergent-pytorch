from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ..emergent import Unit
from ..utils import default, fork, join, to_list


class Sin(nn.Module):
    def __init__(self, omega=100):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class MixerModule(nn.Module):
    def __init__(
        self,
        in_rows: int,
        out_rows: int,
        in_cols: int,
        out_cols=None,
        activation: Callable = Sin(),
    ) -> None:
        super().__init__()
        out_cols = default(out_cols, in_cols)
        self.to_cols = nn.Linear(in_cols, out_cols)
        self.to_rows = nn.Linear(in_rows, out_rows)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        x = rearrange(self.to_cols(x), "b n d -> b d n")
        x = self.activation(x)
        x = rearrange(self.to_rows(x), "b d n -> b n d")
        return x


class MixerLink(nn.Module):
    def __init__(self, in_tokens, out_tokens, features=16, activation=Sin()):
        super().__init__()
        self.out_tokens = out_tokens
        in_rows, out_rows = sum(to_list(in_tokens)), sum(to_list(out_tokens))
        self.mixer = MixerModule(
            in_rows=in_rows, out_rows=out_rows, in_cols=features, activation=activation
        )
        self.activation = activation

    def forward(self, *xs):
        x = join(xs)
        y = self.mixer(x)
        y = self.activation(y)
        ys = fork(y, splits=self.out_tokens)
        return ys


class Mixer(Unit):
    def __init__(self, in_tokens, out_tokens, features=16, activation=Sin()):
        super().__init__()
        self.f = MixerLink(in_tokens, out_tokens, features, activation)
        self.f_inv = MixerLink(out_tokens, in_tokens, features, nn.Identity())

    def forward(self, *xs):
        return self.f(*xs)

    def inverse(self, *xs):
        return self.f_inv(*xs)
