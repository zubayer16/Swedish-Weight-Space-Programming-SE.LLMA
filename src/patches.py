"""Exact-budget random, DARE, magnitude and layer-aware sparse patches."""
import math
import numpy as np
import torch
from .models import group_name


def balanced_budgets(capacities, fraction):
    """Uniform group retention, with stable largest-remainder exact rounding."""
    if not 0 <= fraction <= 1 or any(c < 0 for c in capacities):
        raise ValueError('Invalid fraction or capacities')
    capacities = np.asarray(capacities, dtype=np.int64)
    quotas = capacities*fraction
    counts = np.floor(quotas).astype(np.int64)
    remainder = math.floor(int(capacities.sum())*fraction)-int(counts.sum())
    order = np.argsort(-(quotas-counts), kind='stable')
    counts[order[:remainder]] += 1
    return counts.tolist()


def make_balanced_patch(base, delta, fraction):
    """Keep the largest deltas within every layer/component at a common rate.

    Unlike the original layer heuristic, allocation depends only on group sizes,
    never update norms or observed losses. Embeddings and final norm are separate
    groups, as in the original experiment. All unique positions count, including
    zero deltas; ties follow canonical parameter and row-major position order.
    """
    groups = {}
    for name,value in delta.items():
        groups.setdefault(group_name(name), []).append(name)
    capacities = [sum(delta[n].numel() for n in names) for names in groups.values()]
    budgets = balanced_budgets(capacities, fraction)
    patch,group_counts = {},{}
    for (group,names),budget in zip(groups.items(),budgets):
        sizes = [delta[n].numel() for n in names]
        offsets = np.cumsum([0]+sizes)
        values = np.concatenate([delta[n].numpy().reshape(-1) for n in names])
        sparse = np.zeros_like(values)
        positions = largest(np.abs(values),budget)
        sparse[positions] = values[positions]
        for i,name in enumerate(names):
            patch[name] = torch.from_numpy(sparse[offsets[i]:offsets[i+1]].copy()).reshape(delta[name].shape)
        group_counts[group] = {'capacity':len(values),'selected':budget}
    total = sum(capacities)
    selected = sum(budgets)
    return patch, {'selected_parameters':selected,'total_parameters':total,
        'modified_parameters':sum(int(v.count_nonzero()) for v in patch.values()),
        'retained_fraction':selected/total,'delta_l2':math.sqrt(sum(float(v.double().square().sum()) for v in patch.values())),
        'rescale':1.,'group_budgets':group_counts}


def scaled_dense_patch(delta, alpha):
    """Scale every unique update; alpha is strength, never a sparsity budget."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0,1]')
    if not delta or any(not torch.isfinite(value).all() for value in delta.values()):
        raise ValueError('Dense updates must be nonempty and finite')
    return {name:value * alpha for name,value in delta.items()}


def largest(values, k):
    values = np.asarray(values)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == len(values):
        return np.arange(k)
    cutoff = np.partition(values, len(values)-k)[len(values)-k]
    above = np.flatnonzero(values > cutoff)
    ties = np.flatnonzero(values == cutoff)[:k-len(above)]
    return np.concatenate([above, ties])


def allocate(capacities, scores, budget):
    """Capped proportional allocation, then largest remainders; stable ties."""
    caps = np.asarray(capacities, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if (caps.shape != scores.shape or np.any(caps < 0) or not np.isfinite(scores).all()
            or np.any(scores < 0) or not 0 <= budget <= caps.sum()):
        raise ValueError('Invalid allocation capacities, scores, or budget')
    counts = np.zeros_like(caps)
    while counts.sum() < budget:
        room = caps - counts
        active = room > 0
        weights = np.where(active, scores, 0.)
        if weights.sum() == 0:
            weights = room.astype(float)
        quota = (budget-int(counts.sum())) * weights / weights.sum()
        step = np.minimum(np.floor(quota).astype(np.int64), room)
        if step.sum() == 0:
            order = np.argsort(-quota, kind='stable')
            step[order[:min(budget-int(counts.sum()), int(active.sum()))]] = 1
        counts += step
    return counts.tolist()


def make_patch(base, delta, fraction, method, seed=17):
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0,1]')
    if method not in {'random', 'dare', 'magnitude', 'layer', 'layer_random'}:
        raise ValueError(method)
    names = list(delta)
    lengths = [delta[n].numel() for n in names]
    offsets = np.cumsum([0]+lengths)
    values = np.concatenate([delta[n].numpy().reshape(-1) for n in names])
    n = len(values)
    k = math.floor(n*fraction)
    rng = np.random.default_rng(seed)
    selected = np.zeros(n, dtype=bool)
    if method in {'random', 'dare'}:
        selected[rng.choice(n, k, replace=False)] = True
    elif method == 'magnitude':
        selected[largest(np.abs(values), k)] = True
    else:
        groups = {}
        for i, name in enumerate(names):
            groups.setdefault(group_name(name), []).append(i)
        indices, scores = [], []
        for members in groups.values():
            idx = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in members])
            indices.append(idx)
            d2 = sum(float(delta[names[i]].double().square().sum()) for i in members)
            b2 = sum(float(base[names[i]].double().square().sum()) for i in members)
            scores.append(math.sqrt(d2) / max(math.sqrt(b2), 1e-12))
        counts = allocate([len(i) for i in indices], scores, k)
        for idx, count in zip(indices, counts):
            local = (rng.choice(len(idx), count, replace=False) if method == 'layer_random'
                     else largest(np.abs(values[idx]), count))
            selected[idx[local]] = True
    scale = n/k if method == 'dare' and k else 1.
    sparse = np.where(selected, values*scale, 0).astype(np.float32)
    patch = {name: torch.from_numpy(sparse[offsets[i]:offsets[i+1]].copy()).reshape(delta[name].shape)
             for i, name in enumerate(names)}
    return patch, {'selected_parameters': int(selected.sum()), 'modified_parameters': int(np.count_nonzero(sparse)),
                   'total_parameters': n, 'retained_fraction': k/n, 'delta_l2': float(np.linalg.norm(sparse)),
                   'rescale': scale}
