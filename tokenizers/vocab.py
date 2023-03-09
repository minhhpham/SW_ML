import itertools
from torchtext.vocab import Vocab, build_vocab_from_iterator


DNA_CHARS = ["A", "C", "G", "T"]
# the special padding token for shorter sequences
PAD_TOKEN = "NNN"


def generate_words(length: int, chars=DNA_CHARS):
    """generat all words of fixed length given a character set
    """
    for item in itertools.product(chars, repeat=length):
        yield "".join(item)


def create_NGram_vocab(N=4) -> Vocab:
    words = []
    for sublen in range(N, 0, -1):
        sub_words = generate_words(sublen)
        sub_words = [w + "N"*(N-sublen) for w in sub_words]
        words.extend(sub_words)
    return build_vocab_from_iterator([words], specials=[PAD_TOKEN])
