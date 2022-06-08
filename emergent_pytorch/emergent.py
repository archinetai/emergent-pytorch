import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def to_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def detach(val):
    return val.detach() if isinstance(val, Tensor) else val


def detach_args(*args):
    args = tuple(detach(v) for v in args)
    return args


class InvertibleModule(nn.Module):
    def __init__(self, reverse: bool = False):
        super().__init__()
        self.reverse = reverse

    def __call__(self, *args, **kwargs):
        forward_fn = self.inverse if self.reverse else self.forward
        return forward_fn(*args, **kwargs)

    def __invert__(self):
        return self.inv()

    def inv(self):
        reverse_unit = copy.copy(self)
        reverse_unit.reverse = not self.reverse
        return reverse_unit

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def inverse(self, *args, **kwargs):
        raise NotImplementedError()


class Unit(InvertibleModule):
    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.recon: Optional[Union[Tensor, Tuple[Tensor]]] = None

    def __call__(self, *args):
        forward_fn = self.inverse if self.reverse else self.forward
        if self.training:
            args = detach_args(*args)
            ys = forward_fn(*args)
            self.step(xs=args, ys=to_tuple(ys))
        else:
            ys = forward_fn(*args)
        return ys

    def __neg__(self) -> Optional[Union[Tensor, Tuple[Tensor]]]:
        return self.recon

    def step(self, xs: Tuple[Tensor], ys: Tuple[Tensor]) -> None:
        # Reconstruct input
        inverse_fn = self.forward if self.reverse else self.inverse
        self.recon = inverse_fn(*ys)
        zs = to_tuple(self.recon)
        # Compute loss mean
        loss_fn = self.get_loss_fn()
        losses = [loss_fn(x, z) for x, z in zip(xs, zs)]
        loss = torch.stack(losses).mean()
        # Update model
        optimizer = self.get_optimizer()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Notify hook
        self.on_step(loss)

    def on_step(self, loss: Tensor) -> None:
        """Hook proving the loss during forward learning."""

    def get_loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
