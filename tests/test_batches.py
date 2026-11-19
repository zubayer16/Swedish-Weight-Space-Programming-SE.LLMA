import torch
from transformers import LlamaConfig,LlamaForCausalLM
from src.evaluation import evaluate_batched,verify_equivalence


class Tokens:
    def encode(self,text,add_special_tokens=False):
        assert add_special_tokens is False
        return [int(s) for s in text.split()]


def test_batching_matches_original_causal_model_without_context_leakage():
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
        attention_dropout=0.,attn_implementation='eager'))
    docs = [{'id':str(i),'text':' '.join(str(j%32) for j in range(n))} for i,n in enumerate([9,25,13,31])]
    report = verify_equivalence(model,Tokens(),docs,8,3)
    assert report['passed']
    assert evaluate_batched(model,Tokens(),docs,8,3)['tokens'] == sum([8,24,12,30])
    # Reordering unrelated documents cannot alter the context of an example.
    a = evaluate_batched(model,Tokens(),docs,8,3)['per_document']
    b = evaluate_batched(model,Tokens(),docs[::-1],8,3)['per_document'][::-1]
    assert all(abs(x['loss_sum']-y['loss_sum'])/x['tokens'] < 1e-5 for x,y in zip(a,b))
