from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

from inference.metrics import ClassificationMetrics, ClassMetricsTracker


def evaluate_model(
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    num_classes: int = None
) -> Tuple[float, ClassificationMetrics]:
    """calculate y_hat and calculate average loss per sample against y

    Args:
        batch_iter (Iterable[Tuple[Tensor, Tensor, Tensor]]): data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])
        num_classes: number of classes (None if regression)

    Returns: Tuple of
        average loss per sample,
        average classification metrics (None if not provided)
    """
    total_loss = 0
    n_samples_processed = 0
    if num_classes is not None:
        class_metrics = ClassMetricsTracker(num_classes)
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
            if num_classes is not None:
                class_metrics.add_new_batch(y_hat, y)
            n_samples_processed += y.shape[0]

    avg_class_metrics = None if num_classes is None \
        else class_metrics.get_average_metrics()
    return (
        total_loss / n_samples_processed,
        avg_class_metrics
    )
