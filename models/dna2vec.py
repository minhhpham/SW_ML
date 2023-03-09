import pickle
from typing import Dict, List

import torch
from numpy import ndarray
from torch import Tensor
from torchtext.vocab import Vocab


class DNA2vecAdapter:
    # data: k -> Dict( kmer -> vec )
    data: Dict[int, Dict[str, Tensor]] = None
    vec_size: int = None  # vector size

    def __init__(self, filepath="models/dna2vec_output.pkl"):
        with open(filepath, "rb") as file:
            data: Dict[int, Dict[str, ndarray]] = pickle.load(file)
        self.data: Dict[int, Dict[str, Tensor]] = {}
        for k in data.keys():
            self.data[k] = {
                kmer: torch.from_numpy(data[k][kmer])
                for kmer in data[k].keys()
            }
        self.vec_size = self.data[4]["AAAA"].shape[0]

    def get_vec(self, tokens: List[str]) -> List[Tensor]:
        """convert a list of token into List of Tensor vectors
        token not found in dna2vec get a zero vectors
        """
        out = []
        for token in tokens:
            k = len(token)
            assert k in self.data.keys(), f"{k} not found in dna2vec"
            if token not in self.data[k]:
                out.append(torch.zeros(self.vec_size))
            else:
                out.append(self.data[k][token])
        return out

    def get_embeddings(self, vocab: Vocab) -> Tensor:
        """create a 2D Tensor embedding for a vocab object
        """
        tokens = vocab.get_itos()
        vecs = self.get_vec(tokens)
        return torch.stack(vecs)
