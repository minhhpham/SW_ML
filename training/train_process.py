import time
from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

import settings
from inference.inference import evaluate_model
from inference.metrics import ClassificationMetrics, ClassMetricsTracker
from models.Transformer import calculate_mask
from training.batch import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    epoch_id: int,
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accum_iter=1,
    verbose_freq=40,
    num_classes: int = None,  # None if regression problem
) -> Tuple[float, ClassificationMetrics]:
    """Run all data through a model for a single epoch

    Args:
        batch_iter (Iterable): iterable of data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])
        optimizer (torch.optim.Optimizer)
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate func
        accum_iter (int): per number of iterations to call optimizer.step()
        verbose_freq: per number of batches to print progress
        num_classes: number of classes (None if regression)

    Returns:
        Tuple[float, ClassificationMetrics]:
            average loss per sample and classification metrics
    """
    n_samples_processed = 0
    if num_classes is not None:
        class_metrics_tracker = ClassMetricsTracker(num_classes)
    epoch_loss = 0
    # i = training batch index
    for i, batch in tqdm(enumerate(batch_iter), total=len(batch_iter)):
        x1, x2, y = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        # mask out the blank positions that were padded with index 0
        mask = calculate_mask(x1, x2)
        y_hat = model.forward(x1, x2, mask).squeeze()
        y = y.squeeze()
        loss = loss_func(y_hat, y)
        loss_value = loss.item()

        # compute gradients
        loss.backward()
        # update parameters
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # # update learning rate
        scheduler.step()
        # update total loss across all batches
        epoch_loss += loss_value
        if num_classes is not None:
            batch_metrics = class_metrics_tracker.add_new_batch(y_hat, y)
        n_samples_processed += y.shape[0]
        # update monitor
        monitor.writer.add_scalar(
            "Loss/train/batch",
            loss_value / y.shape[0],
            epoch_id * len(batch_iter) + i
        )
        if num_classes is not None:
            batch_metrics.monitor_add_scalars(
                monitor.writer, step=i, tag_suffix="train/batch")
        del x1, x2, y, mask, loss
    return (
        epoch_loss / n_samples_processed,
        class_metrics_tracker.get_average_metrics() if num_classes else None
    )


def summarize_model(model: nn.Module, x1: Tensor, x2: Tensor):
    """summarize a model into the tensorboard monitor
    """
    mask = calculate_mask(x1, x2)
    monitor.writer.add_graph(model, (x1, x2, mask))


def train_main(
    train_data: Tuple[Tensor, Tensor, Tensor],
    valid_data: Tuple[Tensor, Tensor, Tensor],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    batch_size=64,
    num_epochs=100,
    model_name="Transformer",
    verbose_freq=40,
    num_classes: int = None,  # None if regression problem
) -> nn.Module:
    global monitor
    monitor = settings.TB_MONITOR
    # create data loaders
    x1, x2, y = train_data
    dataloader_train = DataLoader(x1, x2, y, batch_size=batch_size)
    x1, x2, y = valid_data
    dataloader_valid = DataLoader(x1, x2, y, batch_size=batch_size)
    # summarize model, at this point, model and data are both on CPU
    # we use CPU to summarize model to avoid bugs
    summarize_model(model, x1[:100, :], x2[:100, :])
    del x1, x2, y
    # training loop
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"---- Epoch {epoch} Training ----", flush=True)
        start_time = time.time()
        model.train()
        train_loss, class_metrics_epoch = run_epoch(
            epoch_id=epoch,
            batch_iter=dataloader_train,
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            verbose_freq=verbose_freq,
            num_classes=num_classes
        )
        # add train data to monitor
        learn_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        monitor.writer.add_scalar("Loss/train/epoch", train_loss, epoch)
        monitor.writer.add_scalar("Learning rate", learn_rate, epoch)
        monitor.writer.add_scalar(
            "speed/samples_per_sec",
            len(dataloader_train.dataset) / elapsed,
            epoch
        )
        if num_classes is not None:
            class_metrics_epoch.monitor_add_scalars(
                writer=monitor.writer, step=epoch, tag_suffix="train/epoch")
        monitor.writer.flush()

        # save model
        torch.save(model.module.state_dict(),
                   f"saved_models/{model_name}_{epoch}.pt")
        torch.cuda.empty_cache()

        print(f"---- Epoch {epoch} Validation ----", flush=True)
        model.eval()
        valid_loss, class_metrics = evaluate_model(
            batch_iter=dataloader_valid,
            model=model,
            loss_func=loss_func,
            num_classes=num_classes
        )
        # add valid data to monitor
        monitor.writer.add_scalar("Loss/valid", valid_loss, epoch)
        if num_classes is not None:
            class_metrics.monitor_add_scalars(
                writer=monitor.writer, step=epoch, tag_suffix="valid/epoch")
        monitor.writer.flush()
        torch.cuda.empty_cache()
    monitor.writer.close()
    return model.module
