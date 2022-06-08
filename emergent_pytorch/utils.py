from typing import List, Sequence, Union

import torch
from torch import Tensor


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


def is_list(val):
    return isinstance(val, Sequence)


def to_list(val):
    return [val] if not isinstance(val, Sequence) else val


def fork(x: Tensor, splits: Union[List[int], int]):
    return torch.split(x, splits, dim=1) if is_list(splits) else x


def join(xs):
    return torch.cat(xs, dim=1) if is_list(xs) else xs
