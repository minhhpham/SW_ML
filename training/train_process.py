from dataclasses import dataclass
from typing import Iterable, Callable, Tuple
from torch import nn, Tensor
import torch
from inference.inference import evaluate_model
import time
from models.Transformer import Transformer
from training.batch import DataLoader
from torch.utils.tensorboard import SummaryWriter
monitor = SummaryWriter()


@dataclass
class TrainState:
    """
    Track number of steps and samples processed
    """

    accum_step: int = 0  # number of times we have updated parameters
    epoch_step: int = 0   # Steps in the current epoch
    epoch_samples: int = 0  # total # of examples processed in an epoch
    epoch_loss: float = 0  # total loss in an epoch
    epoch_start: float = None  # time we started an epoch (posix seconds)


def run_epoch(
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accum_iter=1,
    train_state=TrainState(),
    verbose_freq=40,
    gpu_id=0,
) -> Tuple[float, TrainState]:
    """Run all data through a model for a single epoch

    Args:
        batch_iter (Iterable): iterable of data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])
        optimizer (torch.optim.Optimizer)
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate func
        accum_iter (int): per number of iterations to call optimizer.step()
        train_state (TrainState)
        verbose_freq: per number of batches to print progress

    Returns:
        Tuple[float, TrainState]: average loss per sample, train state
    """
    train_state.epoch_samples = 0
    train_state.epoch_step = 0
    train_state.epoch_loss = 0
    n_samples_processed = 0
    for i, batch in enumerate(batch_iter):
        x1, x2, y = batch
        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda()
        # mask out the blank positions that were padded with index 0
        mask = (
            torch.logical_or(x1 != 0, x2 != 0)
        ).unsqueeze(-2)
        mask = mask.cuda()
        y_hat = model.forward(x1, x2, mask)
        loss = loss_func(y_hat, y)
        loss_value = loss.item()
        # compute gradients
        loss.backward()
        train_state.epoch_step += 1
        train_state.epoch_samples += y.shape[0]
        # update parameters
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_state.accum_step += 1
        # # update learning rate
        scheduler.step()
        # update total loss across all batches
        train_state.epoch_loss += loss_value
        n_samples_processed += y.shape[0]
        # update monitor
        monitor.add_scalar("Loss/train/batch", loss_value / y.shape[0], i)
        del x1, x2, y, mask, loss
    return train_state.epoch_loss / n_samples_processed, train_state


def train_main(
    train_data: Tuple[Tensor, Tensor, Tensor],
    valid_data: Tuple[Tensor, Tensor, Tensor],
    model: Transformer,
    loss_func: Callable[[Tensor, Tensor], float],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    batch_size=64,
    gpu_id=0,
    num_epochs=100,
    model_name="Transformer",
    verbose_freq=40,
):
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    # create data loaders
    x1, x2, y = train_data
    dataloader_train = DataLoader(x1, x2, y, batch_size=batch_size)
    x1, x2, y = valid_data
    dataloader_valid = DataLoader(x1, x2, y, batch_size=batch_size)
    del x1, x2, y
    # training loop
    train_state = TrainState()
    for epoch in range(num_epochs):
        print(f"----[GPU {gpu_id}] Epoch {epoch} Training ----", flush=True)
        start_time = time.time()
        model.train()
        train_loss, train_state = run_epoch(
            batch_iter=dataloader_train,
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            verbose_freq=verbose_freq
        )
        # add train data to monitor
        learn_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        monitor.add_scalar("Loss/train", train_loss, epoch)
        monitor.add_scalar("Learning rate", learn_rate, epoch)
        monitor.add_scalar(
            "samples/sec",
            len(dataloader_train.dataset) / elapsed,
            epoch
        )
        monitor.flush()

        # save model
        torch.save(model.state_dict(), f"saved_models/{model_name}_{epoch}.pt")
        torch.cuda.empty_cache()

        print(f"----[GPU {gpu_id}] Epoch {epoch} Validation ----", flush=True)
        model.eval()
        valid_loss = evaluate_model(
            batch_iter=dataloader_valid,
            model=model,
            loss_func=loss_func
        )
        # add valid data to monitor
        monitor.add_scalar("Loss/valid", valid_loss, epoch)
        monitor.flush()
        torch.cuda.empty_cache()
    monitor.close
