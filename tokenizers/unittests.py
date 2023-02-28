# modify PYTHONPATH to execute this script in a subdir
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from NGram import nGram_tokenize  # noqa
from vocab import create_NGram_vocab  # noqa
from data import loader  # noqa


def test1() -> None:
    vocab_4gram = create_NGram_vocab(4)
    data = loader.read_data("/data/minhpham/SW-ML-data/SRR622461", limit=100)
    for seq1, seq2, _ in data:
        tokens = nGram_tokenize(seq1, 4)
        indices = vocab_4gram.lookup_indices(tokens)
        print(seq1)
        print("    tokens : ", tokens)
        print("    indices: ", indices)

        tokens = nGram_tokenize(seq2, 4)
        indices = vocab_4gram.lookup_indices(tokens)
        print(seq2)
        print("    tokens : ", tokens)
        print("    indices: ", indices)


if __name__ == "__main__":
    test1()
