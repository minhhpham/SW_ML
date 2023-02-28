from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def rate(
    step: int,
    model_size: int,
    factor: int,
    warmup: int
) -> float:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def create_lr_scheduler(
    optimizer: Optimizer,
    model_size: int,
    factor: int,
    warmup: int
) -> LambdaLR:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """

    def lr_lambda(step):
        return factor * \
            (model_size ** (-0.5) *
                min(step ** (-0.5), step * warmup ** (-1.5))
             )

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step,
            model_size=model_size,
            factor=factor,
            warmup=warmup
        ),
    )
