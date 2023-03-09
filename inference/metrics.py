from dataclasses import dataclass
from typing import List
import torch

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC, Accuracy, F1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_score: float
    auc_roc: float

    def monitor_add_scalars(
            self, writer: SummaryWriter, step: int, tag_suffix="") -> None:
        writer.add_scalar(f"accuracy/{tag_suffix}", self.accuracy, step)
        writer.add_scalar(f"f1_score/{tag_suffix}", self.f1_score, step)
        writer.add_scalar(f"auc_roc/{tag_suffix}", self.auc_roc, step)
        writer.flush()


class ClassMetricsTracker:
    """keep classification metrics for batches
    """

    metrics: List[ClassificationMetrics]
    acc_calculator: Accuracy
    f1_calculator: F1Score
    auc_calculator: AUROC

    def __init__(self, num_classes: int) -> None:
        assert num_classes >= 2, f"invalid number of classes (f{num_classes})"
        self.acc_calculator = \
            Accuracy("multiclass", num_classes=num_classes).to(device)
        self.f1_calculator = \
            F1Score("multiclass", num_classes=num_classes).to(device)
        self.auc_calculator = \
            AUROC("multiclass", num_classes=num_classes).to(device)
        self.metrics = []

    def add_new_batch(self, y_hat: Tensor, y: Tensor) -> ClassificationMetrics:
        """add metrics for a new batch

        Args:
            y_hat (Tensor): prediction
            y (Tensor): ground truth
        Returns:
            ClassificationMetrics: calculated metrics
        """
        metrics = ClassificationMetrics(
            accuracy=self.acc_calculator(y_hat, y),
            f1_score=self.f1_calculator(y_hat, y),
            auc_roc=self.auc_calculator(y_hat, y),
        )
        self.metrics.append(metrics)
        return metrics

    def get_average_metrics(self) -> ClassificationMetrics:
        """get average metrics across all batches

        Returns:
            ClassificationMetrics
        """
        def avg(list_): return sum(list_) / len(list_)
        return ClassificationMetrics(
            accuracy=avg([m.accuracy for m in self.metrics]),
            f1_score=avg([m.f1_score for m in self.metrics]),
            auc_roc=avg([m.auc_roc for m in self.metrics]),
        )
