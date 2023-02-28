from typing import List
from math import ceil


def nGram_tokenize(seq: str, N=4) -> List[str]:
    """tokenize a sequence with N-grams
    if sequence length is not divisible by N, pad with letters "N"
    """
    lseq = len(seq)
    n_tokens = ceil(lseq / N)
    # pad with letter N
    seq_padded = seq + "N"*(n_tokens*N - lseq)
    return [seq_padded[i*N: (i+1)*N] for i in range(n_tokens)]
