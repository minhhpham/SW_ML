# modify PYTHONPATH to execute this script in a subdir
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from data import loader  # noqa
from tokenizers.vocab import create_NGram_vocab  # noqa
from tokenizers.NGram import nGram_tokenize  # noqa
from training.preprocess import batch_load_and_preprocess  # noqa
from training.batch import DataLoader  # noqa
from tqdm import tqdm  # noqa
import torch  # noqa


torch.set_printoptions(threshold=0xffffffff, linewidth=1000)
TEST_BATCH_SIZE = 1024


def test1():
    vocab_4gram = create_NGram_vocab(4)
    x1, x2, y = batch_load_and_preprocess(
        datapath="/data/minhpham/SW-ML-data/SRR622461",
        n_samples_limit=1e7,
        tokenizer=nGram_tokenize,
        vocab=vocab_4gram,
        input_fixed_dim=24
    )
    print(x1.shape)
    print(x2.shape)
    print(y.shape)
    return x1, x2, y


def test2() -> None:
    x1, x2, y = test1()
    dataloader = DataLoader(x1, x2, y, batch_size=TEST_BATCH_SIZE)
    print("checking batches ....")
    for x1, x2, y in tqdm(dataloader):
        assert x1.shape == torch.Size([TEST_BATCH_SIZE, 24]), \
            f"wrong x1 dim: {x1.shape}"
        assert x2.shape == torch.Size([TEST_BATCH_SIZE, 24]), \
            f"wrong x2 dim: {x2.shape}"
        assert y.shape == torch.Size([TEST_BATCH_SIZE, 1]), \
            f"wrong y dim: {y.shape}"


if __name__ == "__main__":
    test2()
