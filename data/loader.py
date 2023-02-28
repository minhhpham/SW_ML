from typing import Iterator, Tuple, List
import os
import io
import snappy
import pickle


def read_data(
    path: str,
    sample_limit=None,
    part_limit=None,
    start_part=0
) -> Iterator[Tuple[str, str, int]]:
    """read data from a path
    return tuples of (seq1, seq2, SW score)
    sample_limit: limit number of samples
    part_limit: limit number of data file part
    """
    part_id = start_part
    part_limit = -1 if part_limit is None else part_limit
    sample_count = 0
    part_count = 0
    filepath = f"{path}_{part_id}"
    inmem_file = io.BytesIO()
    print("loading data ", path)
    while os.path.exists(filepath):
        print("    loading file ", filepath)
        with open(filepath, "rb") as infile:
            snappy.stream_decompress(infile, inmem_file)
            data: List[Tuple[str, List[Tuple[str, int]]]] = pickle.loads(
                inmem_file.getvalue()
            )
            for seq1, alt in data:
                for seq2, score in alt:
                    if sample_count < sample_limit:
                        sample_count += 1
                        yield seq1, seq2, score
        if sample_count == sample_limit:
            break
        # if not reached limits, continue to next file
        part_id += 1
        part_count += 1
        if part_count == part_limit:
            break
        filepath = f"{path}_{part_id}"


def main():
    data = read_data("/data/minhpham/SW-ML-data/SRR622461", limit=100000000)
    for seq1, seq2, score in data:
        # print(seq1, seq2, score)
        pass


if __name__ == "__main__":
    main()
