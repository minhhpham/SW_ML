from typing import Iterable, Callable, Tuple
from torch import nn, Tensor
import torch
from inference.inference import evaluate_model
import time
from tqdm import tqdm
from models.Transformer import Transformer, calculate_mask
from training.batch import DataLoader
from torch.utils.tensorboard import SummaryWriter
monitor = SummaryWriter()


def run_epoch(
    epoch_id: int,
    batch_iter: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accum_iter=1,
    verbose_freq=40,
    gpu_id=0,
) -> Tuple[float]:
    """Run all data through a model for a single epoch

    Args:
        batch_iter (Iterable): iterable of data batches
        model (nn.Module): (x1, x2, mask) -> y
        loss_func (Callable[[Tensor, Tensor], float])
        optimizer (torch.optim.Optimizer)
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate func
        accum_iter (int): per number of iterations to call optimizer.step()
        verbose_freq: per number of batches to print progress

    Returns:
        Tuple[float, TrainState]: average loss per sample
    """
    n_samples_processed = 0
    epoch_loss = 0
    for i, batch in tqdm(enumerate(batch_iter), total=len(batch_iter)):
        x1, x2, y = batch
        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda()
        # mask out the blank positions that were padded with index 0
        mask = calculate_mask(x1, x2)
        mask = mask.cuda()
        y_hat = model.forward(x1, x2, mask)
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
        n_samples_processed += y.shape[0]
        # update monitor
        monitor.add_scalar(
            "Loss/train/batch",
            loss_value / y.shape[0],
            epoch_id * len(batch_iter) + i
        )
        del x1, x2, y, mask, loss
    return epoch_loss / n_samples_processed


def summarize_model(model: nn.Module, x1: Tensor, x2: Tensor):
    """summarize a model into the tensorboard monitor
    """
    mask = calculate_mask(x1, x2)
    monitor.add_graph(model, (x1, x2, mask))


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
    # summarize model
    summarize_model(model, x1[:100, :].cuda(), x2[:100, :].cuda())
    del x1, x2, y
    # training loop
    for epoch in range(num_epochs):
        print(f"----[GPU {gpu_id}] Epoch {epoch} Training ----", flush=True)
        start_time = time.time()
        model.train()
        train_loss = run_epoch(
            epoch_id=epoch,
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
        monitor.add_scalar("Loss/train/epoch", train_loss, epoch)
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
    monitor.close()
