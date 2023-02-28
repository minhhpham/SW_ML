import torch
from torch import Tensor
from typing import Tuple


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x1: Tensor, x2: Tensor, y: Tensor):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.x1[index, :],
            self.x2[index, :],
            self.y[index, :],
        )


def DataLoader(
    x1: Tensor,
    x2: Tensor,
    y: Tensor = None,
    batch_size: int = 64,
    shuffle=True,
    drop_last=True,
) -> torch.utils.data.DataLoader:
    if y is None:
        y = torch.zeros(x1.shape[0], 1)
    dataset = Dataset(x1, x2, y)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        drop_last=drop_last,
        pin_memory=True
    )
