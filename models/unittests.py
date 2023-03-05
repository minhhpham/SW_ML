# modify PYTHONPATH to execute this script in a subdir
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.Transformer import TransformerClassifier, calculate_mask  # noqa
import torch  # noqa
from tokenizers.vocab import create_NGram_vocab  # noqa
from tokenizers.NGram import nGram_tokenize  # noqa
from training.preprocess import batch_load_and_preprocess  # noqa

torch.set_printoptions(linewidth=1000, edgeitems=4)


def test1() -> None:
    """
    perform forward pass of an untrained Transformer model
    """
    n_samples = 100
    input_dim = 24
    vocab_4gram = create_NGram_vocab(4)
    x1, x2, y = batch_load_and_preprocess(
        datapath="/data/minhpham/SW-ML-data/SRR622461",
        n_samples_limit=n_samples,
        tokenizer=nGram_tokenize,
        vocab=vocab_4gram,
        input_fixed_dim=input_dim
    )
    print("x1: ", x1.shape)
    print("x2: ", x2.shape)
    print("y : ", y.shape)

    model = TransformerClassifier(
        vocab_size=len(vocab_4gram),
        stack_size=4,
        d_model=512,
        d_feed_fwd=2048,
        n_attn_heads=8,
        dropout=0.1,
        n_out_classes=2
    )
    # mask out positions where x1 or x2 are not padded
    mask = calculate_mask(x1, x2)
    y_hat = model(x1, x2, mask)
    print("y_hat: ", y_hat.shape)
    print(y_hat)


if __name__ == "__main__":
    test1()
