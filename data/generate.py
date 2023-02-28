from skbio.alignment import StripedSmithWaterman
from random import randrange, choice
from typing import Tuple, List
from argparse import ArgumentParser
from tqdm import tqdm
import snappy
import pickle
import multiprocessing as mp
import io
from math import ceil


VOCAB = ["A", "C", "G", "T"]
MAX_N_MISMATCH = 40
MAX_N_GAPS = 10
SEQ_MAX_LEN = 96
DATA_BATCH_SIZE = 1_000_000


def SmithWaterman(seq1: str, seq2: str) -> int:
    query = StripedSmithWaterman(
        query_sequence=seq1,
        match_score=1,
        mismatch_score=-1,
        gap_open_penalty=2,
        gap_extend_penalty=2,
        score_only=True
    )
    alignment = query(seq2)
    return alignment.optimal_alignment_score


def permute(seq: str, n_mismatch: int, n_gap: int) -> str:
    """create a mutation of a seq
    return seq up to SEQ_MAX_LEN
    """
    newseq = seq
    # permute with mismatches
    for _ in range(n_mismatch):
        pos = randrange(0, len(newseq))  # permute position
        new_char = choice(list(set(VOCAB) - set([seq[pos]])))  # new char
        newseq = newseq[:pos] + new_char + newseq[pos+1:]
    # permute with gaps
    for _ in range(n_gap):
        insert = choice([True, False])
        pos = randrange(0, len(newseq))  # ins/del position
        if insert:
            new_char = choice(VOCAB)
            newseq = newseq[:pos] + new_char + newseq[pos:]
        else:
            newseq = newseq[:pos] + newseq[pos+1:]
    return newseq[:SEQ_MAX_LEN]


def permute_and_score(seq: str) -> List[Tuple[str, int]]:
    """permute new sequenceS and compute SW scoreS against them"""
    out = []
    for n_gap in range(0, MAX_N_GAPS+1, 4):
        for n_mismatch in range(0, MAX_N_MISMATCH+1, 8):
            newseq = permute(seq, n_mismatch, n_gap)
            score = SmithWaterman(seq, newseq)
            out.append((newseq, score))
    return out


def get_reads(filepath: str, limit=1000) -> List[str]:
    """get up to {limit} reads from a fastq file,
    discard reads with ambiguous base N
    limit length of each read to SEQ_MAX_LEN
    """
    out = []
    count = 0
    print("getting reads ...")
    with open(filepath, "r") as file:
        for i, line in tqdm(enumerate(file), total=limit*4):
            if i % 4 != 1:
                continue
            seq = line.strip()
            if "N" in seq:
                continue
            out.append(seq[:SEQ_MAX_LEN])
            count += 1
            if count == limit:
                break
    return out


def write_output(
        output: List[Tuple[str, List[Tuple[str, int]]]],
        outfile: str) -> None:
    """write data of (seq1, seq2, score) to file"""
    print("writing output ... ")
    n_batch = ceil(len(output) / DATA_BATCH_SIZE)
    for i in range(n_batch):
        print(f"    batch {i} / {n_batch-1}")
        print("        pickling")
        pickled_data = pickle.dumps(
            output[i*DATA_BATCH_SIZE: (i+1)*DATA_BATCH_SIZE]
        )  # returns data as a bytes object
        print("        compressing")
        inmem_file = io.BytesIO(pickled_data)
        with open(f"{outfile}_{i}", "wb") as f:
            snappy.stream_compress(inmem_file, f)


def main_kernel(seq: str) -> Tuple[str, List[Tuple[str, int]]]:
    return (seq, permute_and_score(seq))


def main() -> None:
    arg_parser = ArgumentParser(
        description="generate randomly permuted string from a FASTQ file and compute score"  # noqa
    )
    arg_parser.add_argument("-i", "--input", help="input file")
    arg_parser.add_argument(
        "-o", "--output",
        help="output file name",
        default="out.csv"
    )
    args = arg_parser.parse_args()
    seqs = get_reads(args.input, limit=20_000_000)
    out = []
    print("permuting and scoring ... ")
    with mp.Pool(48) as mpPool:
        out = list(
            tqdm(
                mpPool.imap(main_kernel, seqs),
                total=len(seqs)
            )
        )

    [print(x) for x in out[0][1]]
    print(f"output size: {len(out)} x {len(out[0][1])}")
    write_output(out, args.output)


if __name__ == "__main__":
    main()
