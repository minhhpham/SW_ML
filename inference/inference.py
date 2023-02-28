from typing import Iterable, Tuple, Callable
import torch
from torch import Tensor, nn


def evaluate_model(
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
) -> float:
    """calculate y_hat and calculate average loss per sample against y

    Args:
        batch_iter (Iterable[Tuple[Tensor, Tensor, Tensor]]): data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])

    Returns:
        average loss per sample
    """
    total_loss = 0
    n_samples_processed = 0
    with torch.no_grad():
        for x1, x2, y in batch_iter:
            # mask out the blank positions that were padded with index 0
            mask = (
                torch.logical_or(x1 != 0, x2 != 0)
            ).unsqueeze(-2)
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
            mask = mask.cuda()
            y_hat = model.forward(x1, x2, mask)
            total_loss += loss_func(y_hat, y).item()
            n_samples_processed += y.shape[0]
    return total_loss / n_samples_processed
