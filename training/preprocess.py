from typing import List, Callable, Tuple
from torch import Tensor
import torch
from torchtext.vocab import Vocab
from tokenizers.vocab import PAD_TOKEN
import functools
from data.loader import read_data
import multiprocessing as mp
from contextlib import closing


def preprocess_kernel(
    seq: str,
    out_i: int,
    out_tensor: Tensor,
    vocab: Vocab,
    tokenizer: Callable[str, List[str]],
    input_fixed_dim: int,
) -> None:
    """tokenize and find index of a sequence
    pad and trim to make length = input_fixed_dim
    modify row out_i of out_tensor in place
    """
    tokens = tokenizer(seq)
    tokens = tokens[:input_fixed_dim]  # trim long seqs
    # pad short seqs
    if len(tokens) < input_fixed_dim:
        tokens = tokens + (input_fixed_dim - len(tokens)) * [PAD_TOKEN]
    out_tensor[out_i, :] = torch.tensor(
        vocab.lookup_indices(tokens),
        dtype=torch.int32
    )


def create_tensor(
    seqs: List[str],
    tokenizer: Callable[str, List[str]],
    vocab: Vocab,
    input_fixed_dim=24,
) -> Tensor:
    """create 2D tensors from array of sequences.
    Args:
        seqs: list of sequences
        tokenizer (Callable[str, List[str]]): tokenizer function
        vocab (Vocab): vocab class: token -> index
        input_fixed_dim (int, optional): Fixed dim of the index.
        Shorter seqs will be padded with default index on the vocab

    Returns:
        Tuple[Tensor, Tensor, Tensor]: _description_
    """
    n_samples = len(seqs)
    out = torch.zeros((n_samples, input_fixed_dim), dtype=torch.int32)
    kernel = functools.partial(
        preprocess_kernel,
        vocab=vocab,
        tokenizer=tokenizer,
        input_fixed_dim=input_fixed_dim,
        out_tensor=out
    )
    idx = list(range(n_samples))
    with closing(mp.Pool()) as pool:
        pool.starmap(kernel, zip(seqs, idx))
    return out


def batch_load_and_preprocess(
    datapath: str,
    n_samples_limit: int,
    tokenizer: Callable[str, List[str]],
    vocab: Vocab,
    input_fixed_dim=24,
) -> Tuple[Tensor, Tensor, Tensor]:
    """load data from a path,
    then tokenize and index sequences and prepare tensors

    Args:
        datapath (str): Raw data contains an array of (seq1, seq2, score)
        n_samples_limit (int): number of samples
        tokenizer (Callable[str, List[str]]): tokenizer function
        vocab (Vocab): vocab class: token -> index
        input_fixed_dim (int, optional): Fixed dim of the index.
        Shorter seqs will be padded with default index on the vocab

    Returns:
        Tuple[Tensor, Tensor, Tensor]: seq1 tensor, seq2 tensor, score
    """
    x1_out = None
    x2_out = None
    y_out = None
    part_id = 0
    sample_count = 0
    while sample_count < n_samples_limit:
        remain_limit = n_samples_limit - sample_count
        data = list(
            read_data(
                path=datapath,
                sample_limit=remain_limit,
                start_part=part_id,
                part_limit=1
            )
        )
        if len(data) == 0:
            break
        seq1s = [d[0] for d in data]
        seq2s = [d[1] for d in data]
        scores = [d[2] for d in data]
        del data
        print(f"preprocessing batch {part_id} ...")
        print("    scores ....")
        y = torch.tensor(scores, dtype=torch.float)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_out = y if y_out is None \
            else torch.concat([y_out, y])
        print("    seq1s ....")
        x1_tmp = create_tensor(
            seqs=seq1s,
            tokenizer=tokenizer,
            vocab=vocab
        )
        x1_out = x1_tmp if x1_out is None \
            else torch.concat([x1_out, x1_tmp])
        print("    seq2s ....")
        x2_tmp = create_tensor(
            seqs=seq2s,
            tokenizer=tokenizer,
            vocab=vocab
        )
        x2_out = x2_tmp if x2_out is None \
            else torch.concat([x2_out, x2_tmp])

        part_id += 1
        sample_count += y.shape[0]
        print(f"finished processing {sample_count:,} samples")

    return (x1_out, x2_out, y_out)
