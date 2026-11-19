from types import SimpleNamespace
import torch
from src.models import verify
from src.models import compute_delta
import pytest


class Model(torch.nn.Linear):
    def __init__(self):
        super().__init__(2,2)
        self.config = SimpleNamespace(model_type='tiny',vocab_size=2)


class Tokenizer:
    special_tokens_map = {}
    all_special_ids = []
    def get_vocab(self):
        return {'a':0,'b':1}


def test_token_mapping_and_nonfinite_rejected():
    a,b = Model(),Model()
    t,u = Tokenizer(),Tokenizer()
    assert verify(a,b,t,u,{})['compatible']
    u.get_vocab = lambda: {'a':1,'b':0}
    assert not verify(a,b,t,u,{})['compatible']
    u = Tokenizer()
    with torch.no_grad():
        b.weight[0,0] = float('nan')
    assert not verify(a,b,t,u,{})['compatible']
    with pytest.raises(ValueError):
        compute_delta({'a':torch.ones(2)},{'a':torch.ones(3)})
