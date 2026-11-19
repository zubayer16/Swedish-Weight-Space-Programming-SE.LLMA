from types import SimpleNamespace
import math
import pytest
import torch
from src.evaluation import evaluate,paired_interval


class Uniform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
    def forward(self,input_ids,use_cache=False):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape,5,device=input_ids.device))


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        assert add_special_tokens is False
        return [int(x) for x in text.split()]


def test_likelihood_shift_chunk_and_weighting():
    docs = [{'id':'a','text':'0 1 2 3 4 0 1'},{'id':'b','text':'0 1'}]
    result = evaluate(Uniform(),Tokenizer(),docs,3)
    assert result['tokens'] == 7
    assert result['examples'] == 4
    assert result['nll'] == pytest.approx(math.log(5))
    assert paired_interval(result['per_document'],result['per_document']) == [0,0]


def test_no_empty_target():
    with pytest.raises(ValueError):
        evaluate(Uniform(),Tokenizer(),[{'id':'a','text':'0'}])


def test_correct_next_token_targets():
    class Next(Uniform):
        def forward(self,input_ids,use_cache=False):
            logits = torch.zeros(*input_ids.shape,5)
            logits.scatter_(2,((input_ids+1)%5).unsqueeze(-1),10.)
            return SimpleNamespace(logits=logits)
    result = evaluate(Next(),Tokenizer(),[{'id':'x','text':'0 1 2 3 4 0 1'}],3)
    assert result['tokens'] == 6
    assert result['nll'] < .001
