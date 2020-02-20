import torch as th
from typing import Union, List


def th_enforce_symmetry(batch: List[th.Tensor], label: th.Tensor, anti=True):
    anti_batch = [b.flip(1) for b in batch]
    new_label = 1 - label if anti else label
    return anti_batch, new_label


# def th_enforce_symmetry(batch, label):
#     label = label
#     anti_batch = batch.flip(1)
#     return anti_batch, label
