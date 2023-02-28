import torch
from training.preprocess import batch_load_and_preprocess
from tokenizers.vocab import create_NGram_vocab
from tokenizers.NGram import nGram_tokenize
from models.Transformer import Transformer
from training.learning_rate import create_lr_scheduler
from training import train_process


N_SAMPLES = 1e8
INPUT_DIM = 24
MODEL_SIZE = 512
DATA_BATCH_SIZE = 2048


def main() -> None:
    # load data
    vocab_4gram = create_NGram_vocab(4)
    x1, x2, y = batch_load_and_preprocess(
        datapath="/data/minhpham/SW-ML-data/SRR622461",
        n_samples_limit=N_SAMPLES,
        tokenizer=nGram_tokenize,
        vocab=vocab_4gram,
        input_fixed_dim=INPUT_DIM
    )
    # split train, validation, test 80/10/10
    valid_start = int(N_SAMPLES * 0.8)
    test_start = int(N_SAMPLES * 0.9)
    x1_train = x1[:valid_start]
    x2_train = x2[:valid_start]
    y_train = y[:valid_start]
    x1_valid = x1[valid_start:test_start]
    x2_valid = x2[valid_start:test_start]
    y_valid = y[valid_start:test_start]
    x1_test = x1[test_start:]
    x2_test = x2[test_start:]
    y_test = y[test_start:]
    print("train shape: ", x1_train.shape, x2_train.shape, y_train.shape)
    print("valid shape: ", x1_valid.shape, x2_valid.shape, y_valid.shape)
    print("test shape: ", x1_test.shape, x2_test.shape, y_test.shape)

    # model
    model = Transformer(
        vocab_size=len(vocab_4gram),
        stack_size=4,
        d_model=MODEL_SIZE,
        d_feed_fwd=2048,
        n_attn_heads=8,
        dropout=0.1
    )
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        model_size=MODEL_SIZE,
        factor=1.0,
        warmup=400
    )

    # train
    train_process.train_main(
        train_data=(x1_train, x2_train, y_train),
        valid_data=(x1_valid, x2_valid, y_valid),
        model=model,
        loss_func=torch.nn.MSELoss(reduction="sum"),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=DATA_BATCH_SIZE,
        verbose_freq=100
    )

    # test


if __name__ == "__main__":
    main()
