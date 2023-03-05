from typing import Iterable, Tuple, Callable
import torch
from torch import Tensor, nn
from tqdm import tqdm


def count_accurate(y_hat: Tensor, y: Tensor) -> float:
    """Compute the number of correct predictions on a classifier problem."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # The output is a vector -- not just a scalar
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return int(cmp.type(y.dtype).sum())


def evaluate_model(
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    classifying=False
) -> Tuple[float, float]:
    """calculate y_hat and calculate average loss per sample against y

    Args:
        batch_iter (Iterable[Tuple[Tensor, Tensor, Tensor]]): data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])

    Returns:
        average loss per sample, accuracy if classifying else None
    """
    total_loss = 0
    n_samples_processed = 0
    total_accu_cnt = 0 if classifying else None
    with torch.no_grad():
        for x1, x2, y in tqdm(batch_iter):
            # mask out the blank positions that were padded with index 0
            mask = (
                torch.logical_or(x1 != 0, x2 != 0)
            ).unsqueeze(-2)
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
            mask = mask.cuda()
            y_hat = model.forward(x1, x2, mask)
            y_hat = y_hat.squeeze()
            y = y.squeeze()
            total_loss += loss_func(y_hat, y).item()
            if classifying:
                total_accu_cnt += count_accurate(y_hat, y)
            n_samples_processed += y.shape[0]

    return (
        total_loss / n_samples_processed,
        total_accu_cnt / n_samples_processed if classifying else None
    )
