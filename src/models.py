"""Checkpoint compatibility, unique deltas, patch application and weight statistics."""
import torch
import json
import re
import math
import pandas as pd


def snapshot(model):
    return {n: p.detach().cpu().float().clone() for n, p in model.named_parameters()}


def compute_delta(base, adapted):
    if base.keys() != adapted.keys():
        raise ValueError('Parameter names differ')
    result = {}
    for name in base:
        if base[name].shape != adapted[name].shape:
            raise ValueError(f'Shape mismatch: {name}')
        result[name] = adapted[name].float() - base[name].float()
        if not torch.isfinite(result[name]).all():
            raise ValueError(f'Nonfinite delta: {name}')
    return result


@torch.no_grad()
def apply_patch(model, base, patch):
    params = dict(model.named_parameters())
    if params.keys() != base.keys() or patch.keys() != base.keys():
        raise ValueError('Unexpected parameter names')
    for name, p in params.items():
        if patch[name].shape != p.shape:
            raise ValueError(f'Unexpected shape: {name}')
        p.copy_((base[name] + patch[name]).to(p.device))


def aliases(model):
    groups = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        groups.setdefault((p.data_ptr(), p.storage_offset(), tuple(p.shape)), []).append(n)
    return sorted([sorted(v) for v in groups.values() if len(v) > 1])


def verify(base, adapted, tokenizer, other_tokenizer, revisions):
    a, b = dict(base.named_parameters()), dict(adapted.named_parameters())
    fields = ['model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
              'num_key_value_heads','vocab_size','rope_theta','rope_scaling','max_position_embeddings',
              'rms_norm_eps','hidden_act','attention_bias','mlp_bias','tie_word_embeddings']
    architecture = {k: [getattr(base.config,k,None),getattr(adapted.config,k,None)] for k in fields}
    checks = dict(architecture=all(x == y for x,y in architecture.values()),
                  names=a.keys() == b.keys(), shapes=a.keys() == b.keys() and all(a[n].shape == b[n].shape for n in a),
                  token_mapping=tokenizer.get_vocab() == other_tokenizer.get_vocab(),
                  tied_aliases=aliases(base) == aliases(adapted),
                  finite=all(torch.isfinite(p).all().item() for p in list(a.values())+list(b.values())))
    special = {k: [getattr(base.config,k,None),getattr(adapted.config,k,None)] for k in ['bos_token_id','eos_token_id','pad_token_id']}
    report = dict(compatible=all(checks.values()), checks=checks, revisions=revisions, architecture=architecture,
                  config_special_tokens=special, tokenizer_special_tokens=[tokenizer.special_tokens_map,other_tokenizer.special_tokens_map],
                  tokenizer_special_ids=[tokenizer.all_special_ids,other_tokenizer.all_special_ids],
                  tied_aliases=aliases(base), precision=[str(next(base.parameters()).dtype),str(next(adapted.parameters()).dtype)],
                  declared_precision=[str(getattr(base.config,'torch_dtype',None)),str(getattr(adapted.config,'torch_dtype',None))],
                  evaluation_policy='Base tokenizer; add_special_tokens=False; no BOS/EOS or padding; first token of each document unscored.')
    return report


def group_name(name):
    match = re.search(r'layers\.(\d+)', name)
    return f'layer.{int(match[1]):02d}' if match else ('embeddings' if 'embed' in name or 'lm_head' in name else 'final_norm')


def analyze(base, delta, output):
    rows = []
    total = sum(float(d.double().square().sum()) for d in delta.values())
    for name, d in delta.items():
        d2 = float(d.double().square().sum())
        b2 = float(base[name].double().square().sum())
        component = ('attention' if 'self_attn' in name else 'mlp' if 'mlp' in name
                     else 'embeddings' if 'embed' in name or 'lm_head' in name else 'normalization')
        rows.append(dict(name=name, group=group_name(name), component=component, parameters=d.numel(),
                         nonzero=int(d.count_nonzero()), delta_l2=math.sqrt(d2), relative_l2=math.sqrt(d2)/max(math.sqrt(b2),1e-12),
                         max_abs=float(d.abs().max()), mean_abs=float(d.abs().mean()), squared_norm=d2,
                         squared_norm_share=d2/total if total else 0.))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'weight_statistics.csv', index=False)
    frame.groupby('component').agg(parameters=('parameters','sum'), nonzero=('nonzero','sum'), squared_norm_share=('squared_norm_share','sum')).to_csv(output/'component_statistics.csv')
    return frame
