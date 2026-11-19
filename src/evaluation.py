"""Reference and batched likelihood scoring with paired document statistics."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import argparse
from itertools import combinations
import json
from pathlib import Path
import pandas as pd


@torch.inference_mode()
def evaluate(model, tokenizer, documents, context_length=128):
    """Each target scored once. Nonoverlapping target blocks share one boundary token.

    No inter-document context; no truncation. All but a document's first token
    are scored. Context resets every context_length-1 targets for all conditions.
    """
    if context_length < 2:
        raise ValueError('context_length must be >=2')
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for doc in documents:
        ids = tokenizer.encode(doc['text'], add_special_tokens=False)
        loss_sum, count, chunks = 0., 0, 0
        for start in range(0, len(ids)-1, context_length-1):
            tokens = torch.tensor([ids[start:start+context_length]], device=device)
            logits = model(input_ids=tokens, use_cache=False).logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1),reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite likelihood')
            loss_sum += loss.item()
            count += tokens.shape[1]-1
            chunks += 1
        if count == 0:
            raise ValueError('Document has fewer than two tokens')
        rows.append(dict(id=doc['id'], loss_sum=loss_sum,tokens=count,chunks=chunks))
    if not rows:
        raise ValueError('Empty evaluation')
    return dict(nll=sum(x['loss_sum'] for x in rows)/sum(x['tokens'] for x in rows),
                tokens=sum(x['tokens'] for x in rows),documents=len(rows),examples=sum(x['chunks'] for x in rows),per_document=rows)


def paired_interval(base, candidate, samples=2000, seed=2026):
    if [r['id'] for r in base] != [r['id'] for r in candidate]:
        raise ValueError('Unpaired documents')
    if [r['tokens'] for r in base] != [r['tokens'] for r in candidate]:
        raise ValueError('Token counts differ')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0,len(base),(samples,len(base)))
    diff = np.array([b['loss_sum']-c['loss_sum'] for b,c in zip(base,candidate)])
    tokens = np.array([b['tokens'] for b in base])
    return np.quantile(diff[idx].sum(1)/tokens[idx].sum(1), [.025,.975]).tolist()


@torch.inference_mode()
def evaluate_batched(model, tokenizer, documents, context_length=128, batch_size=8):
    if batch_size == 1:
        return evaluate(model,tokenizer,documents,context_length)
    if batch_size < 1 or context_length < 2 or not documents:
        raise ValueError('Invalid batch size, context length, or empty documents')
    model.eval()
    device = next(model.parameters()).device
    rows = [dict(id=d['id'],loss_sum=0.,tokens=0,chunks=0) for d in documents]
    pending = defaultdict(list)

    def score(batch):
        inputs = torch.tensor([tokens for _,tokens in batch],device=device)
        logits = model(input_ids=inputs,use_cache=False).logits[:,:-1].float()
        losses = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),inputs[:,1:].reshape(-1),
                                 reduction='none').reshape(len(batch),-1).sum(dim=1)
        if not torch.isfinite(losses).all():
            raise ValueError('Nonfinite likelihood')
        for (doc_index,ids),loss in zip(batch,losses.tolist()):
            rows[doc_index]['loss_sum'] += loss
            rows[doc_index]['tokens'] += len(ids)-1
            rows[doc_index]['chunks'] += 1

    for i,doc in enumerate(documents):
        ids = tokenizer.encode(doc['text'],add_special_tokens=False)
        if len(ids) < 2:
            raise ValueError('Document has fewer than two tokens')
        for start in range(0,len(ids)-1,context_length-1):
            chunk = ids[start:start+context_length]
            queue = pending[len(chunk)]
            queue.append((i,chunk))
            if len(queue) == batch_size:
                score(queue)
                queue.clear()
    for queue in pending.values():
        if queue:
            score(queue)
    return dict(nll=sum(r['loss_sum'] for r in rows)/sum(r['tokens'] for r in rows),
                tokens=sum(r['tokens'] for r in rows),documents=len(rows),
                examples=sum(r['chunks'] for r in rows),per_document=rows)


def verify_equivalence(model,tokenizer,documents,context_length,batch_size):
    scalar = evaluate(model,tokenizer,documents,context_length)
    batched = evaluate_batched(model,tokenizer,documents,context_length,batch_size)
    for name in ['tokens','documents','examples']:
        if scalar[name] != batched[name]:
            raise AssertionError(f'Batching changed {name}')
    max_per_token_error = 0.
    for a,b in zip(scalar['per_document'],batched['per_document']):
        if (a['id'],a['tokens'],a['chunks']) != (b['id'],b['tokens'],b['chunks']):
            raise AssertionError('Batching changed document boundaries/counts')
        error = abs(a['loss_sum']-b['loss_sum'])/a['tokens']
        max_per_token_error = max(max_per_token_error,error)
        if error > 1e-5:
            raise AssertionError(f'Batched NLL differs from reference by {error}')
    return {'passed':True,'max_document_nll_absolute_error':max_per_token_error,'tolerance':1e-5,
            'documents':len(documents),'batch_size':batch_size}


def paired_bootstrap(reference, candidate, samples=2000, seed=2026):
    if not reference or [r['id'] for r in reference] != [r['id'] for r in candidate]:
        raise ValueError('Documents must be nonempty, identically ordered, and paired')
    tokens = np.array([r['tokens'] for r in reference],dtype=np.int64)
    if not np.array_equal(tokens,[r['tokens'] for r in candidate]) or np.any(tokens <= 0):
        raise ValueError('Positive token counts must match')
    delta = np.array([c['loss_sum']-r['loss_sum'] for r,c in zip(reference,candidate)])
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite document losses')
    draws = np.random.default_rng(seed).integers(0,len(tokens),(samples,len(tokens)))
    distribution = delta[draws].sum(1)/tokens[draws].sum(1)
    low,high = np.quantile(distribution,[.025,.975])
    return {'delta_nll':float(delta.sum()/tokens.sum()),'ci_low':float(low),'ci_high':float(high)}


def analyze_results(directory, samples=2000):
    details = json.loads((directory/'document_metrics.json').read_text(encoding='utf-8'))
    rows = pd.read_csv(directory/'experiment_results.csv')
    summaries, comparisons, per_seed = [], [], []
    for language in ['sv','en']:
        metric_key = f'{language}:test'
        base = details['base:0.0:0'][metric_key]['per_document']
        adapted = details['adapted:1.0:0'][metric_key]['per_document']
        tokens = np.array([d['tokens'] for d in base])
        base_losses = np.array([d['loss_sum'] for d in base])
        adapted_losses = np.array([d['loss_sum'] for d in adapted])
        draws = np.random.default_rng(2026).integers(0,len(base),(samples,len(base)))
        denominators = tokens[draws].sum(1)
        full_gain_draws = (base_losses-adapted_losses)[draws].sum(1)/denominators
        full_gain = (base_losses-adapted_losses).sum()/tokens.sum()
        gain_defined = np.quantile(full_gain_draws,.025) > 0
        grouped = {}
        for key, conditions in details.items():
            method,fraction,seed = key.split(':')
            doc = conditions[metric_key]['per_document']
            stats = paired_bootstrap(base,doc,samples)
            per_seed.append(dict(condition=key,method=method,fraction=float(fraction),seed=int(seed),language=language,**stats))
            grouped.setdefault((method,float(fraction)),[]).append(doc)
        mean_docs = {}
        for (method,fraction),replicates in grouped.items():
            # paired_bootstrap above validates all IDs and token counts before stacking.
            loss = np.mean([[d['loss_sum'] for d in docs] for docs in replicates],axis=0)
            doc = [dict(id=b['id'],tokens=b['tokens'],loss_sum=float(v)) for b,v in zip(base,loss)]
            mean_docs[(method,fraction)] = doc
            nll_draws = loss[draws].sum(1)/denominators
            nll_ci = np.quantile(nll_draws,[.025,.975])
            delta_stats = paired_bootstrap(base,doc,samples)
            seed_nll = [sum(d['loss_sum'] for d in docs)/tokens.sum() for docs in replicates]
            retention = (base_losses-loss).sum()/tokens.sum()/full_gain if gain_defined else np.nan
            retention_ci = [np.nan,np.nan]
            if gain_defined and np.all(full_gain_draws > 0):
                retention_ci = np.quantile(((base_losses-loss)[draws].sum(1)/denominators)/full_gain_draws,[.025,.975])
            summaries.append(dict(method=method,fraction=fraction,language=language,nll=float(loss.sum()/tokens.sum()),
                nll_ci_low=float(nll_ci[0]),nll_ci_high=float(nll_ci[1]),delta_nll_vs_base=delta_stats['delta_nll'],
                delta_ci_low=delta_stats['ci_low'],delta_ci_high=delta_stats['ci_high'],
                english_regression=delta_stats['delta_nll'] if language == 'en' else np.nan,
                swedish_gain_retention=retention if language == 'sv' else np.nan,
                retention_ci_low=retention_ci[0] if language == 'sv' else np.nan,
                retention_ci_high=retention_ci[1] if language == 'sv' else np.nan,
                documents=len(base),tokens=int(tokens.sum()),seeds=len(replicates),
                seed_std_nll=float(np.std(seed_nll,ddof=1)) if len(seed_nll)>1 else np.nan))
        for reference,candidate in combinations(mean_docs,2):
            comparisons.append(dict(language=language,reference=reference[0],reference_fraction=reference[1],
                candidate=candidate[0],candidate_fraction=candidate[1],documents=len(base),tokens=int(tokens.sum()),
                **paired_bootstrap(mean_docs[reference],mean_docs[candidate],samples)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory/'method_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(directory/'paired_comparisons.csv',index=False)
    pd.DataFrame(per_seed).to_csv(directory/'paired_vs_base_per_seed.csv',index=False)
    metadata = {'bootstrap_unit':'document','samples':samples,'seed':2026,'confidence':.95,
                'difference_sign':'candidate minus reference; negative favors candidate',
                'random_methods':'fixed-seed mean document loss, then document bootstrap; seed SD separate',
                'multiplicity':'pointwise exploratory intervals; no multiple-comparison correction',
                'gain_retention':'defined only if 95% full-adaptation gain interval is strictly positive; ratio CI omitted if any bootstrap denominator is nonpositive'}
    (directory/'statistics.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return summary


def select_dense_alpha(development):
    """Minimize equal-language mean NLL on development only; smaller-alpha ties."""
    if not development:
        raise ValueError('No development measurements')
    scored = []
    for alpha,metrics in development.items():
        values = [metrics[f'{lang}:test']['nll'] for lang in ['sv','en']]
        if not 0 <= float(alpha) <= 1 or not np.isfinite(values).all():
            raise ValueError('Invalid development alpha or NLL')
        scored.append((float(np.mean(values)),float(alpha)))
    objective,alpha = min(scored)
    return {'alpha':alpha,'objective':objective,
            'rule':'Minimum arithmetic mean of Swedish and English token-weighted development NLL; smaller alpha breaks exact ties'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/wikipedia/full'))
    analyze_results(parser.parse_args().results)


if __name__ == '__main__':
    main()
