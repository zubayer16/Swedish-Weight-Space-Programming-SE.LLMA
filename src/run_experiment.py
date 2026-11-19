"""Reproducible experiment orchestration and completed-result audit."""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml
from .models import snapshot, compute_delta, apply_patch, verify, analyze
from .patches import make_patch, make_balanced_patch
from .evaluation import paired_interval, analyze_results, evaluate_batched, verify_equivalence
from .reporting import plot_weights, plot_metrics, plot_expanded
from .datasets import verify_frozen, digest


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = 'quick' if args.quick else 'full'
    out = root/cfg.get('results_directory','results')/mode
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'status.json',{'status':'running'})
    (out/'figures').mkdir(exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    unit = subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,capture_output=True,text=True)
    (out/'unit_checks.txt').write_text(unit.stdout+unit.stderr,encoding='utf-8')
    print(unit.stdout,flush=True)
    unit.check_returncode()
    api = HfApi()
    revisions = {k:(cfg[k+'_revision'] if len(cfg[k+'_revision']) == 40 else
                    api.model_info(cfg[k+'_model'],revision=cfg[k+'_revision']).sha) for k in ['base','swedish']}
    dump(out/'resolved_config.json',dict(config=cfg,revisions=revisions,mode=mode))
    tokenizers = [AutoTokenizer.from_pretrained(cfg[k+'_model'],revision=revisions[k]) for k in ['base','swedish']]
    models = [AutoModelForCausalLM.from_pretrained(cfg[k+'_model'],revision=revisions[k],torch_dtype=torch.float32,attn_implementation='eager') for k in ['base','swedish']]
    report = verify(*models,*tokenizers,revisions)
    dump(out/'compatibility.json',report)
    if not report['compatible']:
        raise RuntimeError('Checkpoint compatibility failed; see compatibility.json')
    base, adapted = map(snapshot,models)
    delta = compute_delta(base,adapted)
    # Gate actual checkpoint arithmetic before any behavioral measurement.
    for fraction, expected in [(0,base),(1,adapted)]:
        patch,_ = make_patch(base,delta,fraction,'magnitude')
        apply_patch(models[0],base,patch)
        for n,p in models[0].named_parameters():
            torch.testing.assert_close(p.cpu(),expected[n],rtol=1e-5,atol=1e-7)
    dump(out/'validation.json',{'unit_tests_passed':True,'checkpoint_zero_and_full_reconstruction_passed':True,
                               'reconstruction_rtol':1e-5,'reconstruction_atol':1e-7})
    del patch, adapted
    model = models[0]
    del models[1:]
    gc.collect()
    frame = analyze(base,delta,out)
    plot_weights(frame,out/'figures')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    datasets = {}
    hashes = {}
    if cfg.get('evaluation_dataset') != 'wikipedia':
        raise ValueError('This configuration must explicitly select the frozen Wikipedia dataset')
    data_dir = root/cfg['dataset_directory']
    manifest, frozen = verify_frozen(data_dir)
    if manifest['requested_per_language'] != cfg['expected_documents_per_language']:
        raise ValueError('Frozen corpus size does not match evaluation configuration')
    hashes['manifest'] = hashlib.sha256((data_dir/'dataset_manifest.json').read_bytes()).hexdigest()
    for lang in ['sv','en']:
        path = root/'data'/f'sample_{lang}.jsonl'
        hashes[f'{lang}:legacy_dev'] = hashlib.sha256(path.read_bytes()).hexdigest()
        docs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        # Preserve original files and IDs, but treat all previous examples as legacy development.
        datasets[lang,'dev'] = docs[:2] if args.quick else docs
        datasets[lang,'test'] = frozen[lang][:2] if args.quick else frozen[lang]
        hashes[f'{lang}:test'] = manifest['datasets'][lang]['sha256']
    dump(out/'dataset_usage.json',{'dataset':'frozen_wikipedia','hashes':hashes,
         'selection':'first 2 per split for quick; all for full','evaluation_context':cfg['context_length'],
         'document_ids':{f'{lang}:{split}':[d['id'] for d in docs] for (lang,split),docs in datasets.items()}})
    batch_size = cfg.get('inference_batch_size',1)
    check_docs = datasets['sv','test'][:2]+datasets['en','test'][:2]
    dump(out/'batch_equivalence.json',verify_equivalence(model,tokenizers[0],check_docs,cfg['context_length'],batch_size))
    seeds = cfg['seeds'][:1] if args.quick else cfg['seeds']
    conditions = [('base',0.,0),('adapted',1.,0)]
    for fraction in cfg['fractions']:
        for method in ['random','dare','magnitude','layer','layer_random']:
            conditions.extend((method,fraction,s) for s in (seeds if method in ['random','dare','layer_random'] else [0]))
    identity = {'config':cfg,'mode':mode,'data':hashes,'core_source':{
        name:hashlib.sha256((root/'src'/name).read_bytes()).hexdigest() for name in
        ['models.py','patches.py','evaluation.py','datasets.py']}}
    progress_path = out/'condition_progress.json'
    rows, details, completed = [], {}, []
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['identity'] != identity:
            raise ValueError('Cannot resume: configuration, data, or core source changed')
        rows,details,completed = progress['rows'],progress['details'],progress['completed']
    dump(out/'run_identity.json',identity)
    for method,fraction,seed in conditions:
        key = f'{method}:{fraction}:{seed}'
        if key in completed:
            print(f'Reusing completed condition {key}',flush=True)
            continue
        print(f'Evaluating {method} fraction={fraction} seed={seed}',flush=True)
        patch, info = make_patch(base,delta,fraction,'magnitude' if method in ['base','adapted'] else method,seed)
        apply_patch(model,base,patch)
        del patch
        details[key] = {}
        for (lang,split),docs in datasets.items():
            metric = evaluate_batched(model,tokenizers[0],docs,cfg['context_length'],batch_size)
            details[key][f'{lang}:{split}'] = metric
            rows.append(dict(method=method,fraction=fraction,seed=seed,language=lang,split=split,**info,
                             **{k:v for k,v in metric.items() if k != 'per_document'}))
        dump(out/'document_metrics.json',details)
        pd.DataFrame(rows).to_csv(out/'experiment_results.csv',index=False)
        completed.append(key)
        dump(progress_path,dict(identity=identity,rows=rows,details=details,completed=completed))
    result = pd.DataFrame(rows)
    result['gain_retention'] = np.nan
    for (lang,split),docs in datasets.items():
        subset = (result.language == lang) & (result.split == split)
        b = details['base:0.0:0'][f'{lang}:{split}']
        a = details['adapted:1.0:0'][f'{lang}:{split}']
        ci = paired_interval(b['per_document'],a['per_document'],cfg['bootstrap_samples'])
        for idx in result[subset].index:
            row = result.loc[idx]
            key = f'{row.method}:{row.fraction}:{row.seed}'
            c = details[key][f'{lang}:{split}']
            lo,hi = paired_interval(b['per_document'],c['per_document'],cfg['bootstrap_samples'])
            result.loc[idx,'delta_nll_vs_base'] = c['nll']-b['nll']
            result.loc[idx,'gain_ci_low'] = lo
            result.loc[idx,'gain_ci_high'] = hi
            if ci[0] > 0:
                result.loc[idx,'gain_retention'] = (b['nll']-c['nll'])/(b['nll']-a['nll'])
    result.to_csv(out/'experiment_results.csv',index=False)
    summary = analyze_results(out,cfg['bootstrap_samples'])
    plot_expanded(summary,out/'figures')
    result.groupby(['method','fraction','language','split']).agg(mean_nll=('nll','mean'),std_nll=('nll','std'),seeds=('seed','count')).to_csv(out/'seed_summary.csv')
    dump(out/'baseline_metrics.json',{k:v for k,v in details.items() if k.startswith(('base:','adapted:'))})
    dump(out/'environment.json',dict(python=sys.version,platform=platform.platform(),device=device,
         packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','pandas','matplotlib','huggingface-hub','pytest','PyYAML','requests','beautifulsoup4']},
         data_sha256=hashes,seconds=time.time()-started))
    plot_metrics(result,out/'figures')
    dump(out/'status.json',{'status':'complete','conditions':len(conditions)})
    if not args.quick:
        # Publish convenient primary tables only after every configured condition completes.
        for filename in ['experiment_results.csv','method_summary.csv','paired_comparisons.csv']:
            shutil.copyfile(out/filename,root/'results'/filename)
        (root/'results'/'figures').mkdir(exist_ok=True)
        for figure in (out/'figures').glob('*.png'):
            shutil.copyfile(figure,root/'results'/'figures'/figure.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiment.yaml')
    parser.add_argument('--resume',action='store_true',help='Reuse only completed conditions with identical data/config/core source')
    parser.add_argument('--audit',action='store_true',help='Audit completed full results without model inference')
    parser.add_argument('--follow-up',action='store_true',help='Run the prespecified balanced-layer follow-up and export its results')
    parser.add_argument('--dense-follow-up',action='store_true',help='Select dense adaptation strength on development data, then confirm on fresh texts')
    parser.add_argument('--scientific-audit',action='store_true',help='Independently check all saved corpora, metrics, statistics and reports')
    parser.add_argument('--check-weights',action='store_true',help='With --scientific-audit, reconstruct all original masks from cached checkpoints')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick',action='store_true')
    mode.add_argument('--full',action='store_true')
    args = parser.parse_args()
    if args.scientific_audit:
        if args.quick or args.full or args.audit or args.follow_up or args.dense_follow_up:
            parser.error('--scientific-audit is a separate action')
        scientific_audit(args.check_weights)
        return
    if args.check_weights:
        parser.error('--check-weights requires --scientific-audit')
    if args.dense_follow_up:
        if args.quick or args.full or args.audit or args.follow_up:
            parser.error('--dense-follow-up is a separate experiment')
        run_dense_follow_up()
        return
    if args.follow_up:
        if args.quick or args.full or args.audit:
            parser.error('--follow-up is a separate experiment')
        run_follow_up()
        return
    if args.audit:
        if args.quick or args.full:
            parser.error('--audit cannot be combined with --quick or --full')
        audit()
        return
    if not (args.quick or args.full):
        parser.error('Choose --quick, --full or --audit')
    try:
        run(args)
    except Exception as exc:
        cfg = yaml.safe_load(Path(args.config).read_text())
        out = Path(__file__).resolve().parents[1]/cfg.get('results_directory','results')/('quick' if args.quick else 'full')
        out.mkdir(parents=True,exist_ok=True)
        dump(out/'status.json',{'status':'failed','error':repr(exc)})
        raise


def verify_source_lineage(root, recorded_sources):
    """Verify recorded bytes or the exact definitions preserved by consolidation.

    Original run identities are historical evidence and are never rewritten to
    pretend the six-module layout produced an earlier experiment.
    """
    refactor_path = root/'results'/'source_refactor.json'
    refactor = json.loads(refactor_path.read_text(encoding='utf-8')) if refactor_path.exists() else {'files':{}}
    for name,expected in recorded_sources.items():
        path = root/'src'/name
        if path.exists():
            assert digest(path.read_bytes()) == expected, f'Source changed: {name}'
            continue
        entry = refactor['files'].get(name)
        assert entry and entry['original_sha256'] == expected, f'No verified source lineage: {name}'
        tree = ast.parse((root/'src'/entry['module']).read_text(encoding='utf-8'))
        definitions = {node.name:digest(ast.dump(node,include_attributes=False)) for node in tree.body
                       if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        for definition,fingerprint in entry['definitions'].items():
            assert definitions.get(definition) == fingerprint, f'Changed definition: {name}:{definition}'


def audit():
    root = Path(__file__).resolve().parents[1]
    out = root/'results'/'wikipedia'/'full'
    manifest,datasets = verify_frozen(root/'data')
    status = json.loads((out/'status.json').read_text())
    assert status == {'status':'complete','conditions':24}, status
    frame = pd.read_csv(out/'experiment_results.csv')
    assert len(frame) == 96
    assert not frame.duplicated(['method','fraction','seed','language','split']).any()
    assert np.isfinite(frame.nll).all()
    assert (frame.selected_parameters == np.floor(frame.fraction*frame.total_parameters)).all()
    details = json.loads((out/'document_metrics.json').read_text())
    assert len(details) == 24
    counts = {}
    scalar = json.loads((root/'results/wikipedia/scalar_reference/document_metrics.json').read_text())
    max_scalar_error = 0.
    for lang,docs in datasets.items():
        ids = [d['id'] for d in docs]
        reference = details['base:0.0:0'][f'{lang}:test']
        assert reference['documents'] == 150
        expected_tokens = [d['tokens'] for d in reference['per_document']]
        for key,condition in details.items():
            metric = condition[f'{lang}:test']
            assert [r['id'] for r in metric['per_document']] == ids, key
            assert [r['tokens'] for r in metric['per_document']] == expected_tokens, key
            assert metric['tokens'] == sum(expected_tokens)
            assert metric['documents'] == 150
            nll = sum(d['loss_sum'] for d in metric['per_document'])/sum(expected_tokens)
            assert abs(metric['nll']-nll) < 1e-10
        old = scalar['base:0.0:0'][f'{lang}:test']['per_document']
        for a,b in zip(old,reference['per_document']):
            assert (a['id'],a['tokens'],a['chunks']) == (b['id'],b['tokens'],b['chunks'])
            max_scalar_error = max(max_scalar_error,abs(a['loss_sum']-b['loss_sum'])/a['tokens'])
        counts[lang] = {'documents':150,'tokens':sum(expected_tokens)}
    assert max_scalar_error < 1e-5, max_scalar_error
    legacy_env = json.loads((root/'results/full/environment.json').read_text())
    for lang in ['sv','en']:
        assert digest((root/'data'/f'sample_{lang}.jsonl').read_bytes()) == legacy_env['data_sha256'][lang]
    current = json.loads((out/'resolved_config.json').read_text())['config']
    legacy = json.loads((root/'results/full/resolved_config.json').read_text())['config']
    fixed = ['base_model','swedish_model','base_revision','swedish_revision','fractions','seeds','context_length','threads','bootstrap_samples']
    assert all(current[k] == legacy[k] for k in fixed)
    scalar_identity = json.loads((root/'results/wikipedia/scalar_reference/run_identity.json').read_text())
    run_identity = json.loads((out/'run_identity.json').read_text())
    verify_source_lineage(root,scalar_identity['core_source'])
    verify_source_lineage(root,run_identity['core_source'])
    assert digest((root/'data/dataset_manifest.json').read_bytes()) == run_identity['data']['manifest']
    live_config = yaml.safe_load((root/'configs/experiment.yaml').read_text())
    assert live_config == current, 'Current configuration differs from completed results'
    for filename in ['experiment_results.csv','method_summary.csv','paired_comparisons.csv']:
        assert digest((root/'results'/filename).read_bytes()) == digest((out/filename).read_bytes())
    summary = pd.read_csv(out/'method_summary.csv')
    assert len(summary) == 24 and (summary.documents == 150).all()
    assert (summary.delta_ci_low <= summary.delta_ci_high).all()
    report = {'passed':True,'conditions':24,'metric_rows':96,'corpus':counts,
              'all_conditions_identical_document_ids_and_token_counts':True,
              'legacy_text_bytes_and_original_study_settings_unchanged':True,
              'original_core_definitions_unchanged':True,
              'source_verification':'Original byte hashes, or unchanged definition ASTs via results/source_refactor.json',
              'all_300_base_documents_max_scalar_vs_batched_nll_error':max_scalar_error,
              'scalar_agreement_tolerance':1e-5,'manifest_sha256':digest((root/'data/dataset_manifest.json').read_bytes())}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def run_follow_up():
    """Preserve the main study; finish the fixed comparison and export numerical results."""
    from .datasets import collect, USER_AGENT, timestamp
    from .reporting import export_results
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/experiment.yaml').read_text())
    plan = yaml.safe_load((root/'configs/follow_up.yaml').read_text())
    out = root/plan['results_directory']
    out.mkdir(parents=True,exist_ok=True)
    start = time.time()
    dump(out/'process.json',{'pid':os.getpid(),'started_at':timestamp()})
    try:
        # Verify the original evidence before using saved controls.
        audit()
        unit = subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,capture_output=True,text=True)
        (out/'unit_checks.txt').write_text(unit.stdout+unit.stderr,encoding='utf-8')
        unit.check_returncode()
        print(unit.stdout,flush=True)
        identity = {'config':cfg,'plan':plan,'plan_sha256':digest((root/'reports/follow_up_plan.md').read_bytes()),
            'source':{p.name:digest(p.read_bytes()) for p in (root/'src').glob('*.py')},
            'original_artifacts':{name:digest((root/plan['original_results']/name).read_bytes()) for name in
                ['document_metrics.json','experiment_results.csv','method_summary.csv','paired_comparisons.csv']}}
        design_path = out/'design.json'
        if design_path.exists():
            if json.loads(design_path.read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError('Follow-up design/source changed; refusing to mix runs')
        else:
            dump(design_path,{'frozen_before_collection_at':timestamp(),'identity':identity})
        dump(out/'status.json',{'status':'collecting','pid':os.getpid()})
        confirmation_dir = root/plan['confirmation_directory']
        collect(confirmation_dir,plan['confirmation_documents_per_language'],USER_AGENT,5000,root/'data')
        original_manifest,observed = verify_frozen(root/'data')
        new_manifest,fresh = verify_frozen(confirmation_dir)
        old = [doc for docs in observed.values() for doc in docs]
        new = [doc for docs in fresh.values() for doc in docs]
        if ({(d['language'],d['article_id']) for d in old} & {(d['language'],d['article_id']) for d in new}
                or {d['content_sha256'] for d in old} & {d['content_sha256'] for d in new}
                or {d['wikidata_id'] for d in old if d.get('wikidata_id')} & {d['wikidata_id'] for d in new if d.get('wikidata_id')}):
            raise ValueError('Confirmation sample overlaps with previously observed pages/topics/hashes')
        usage = {'original_manifest_sha256':digest((root/'data/dataset_manifest.json').read_bytes()),
            'confirmation_manifest_sha256':digest((confirmation_dir/'dataset_manifest.json').read_bytes()),
            'disjoint_page_ids_topics_and_text_hashes':True,
            'confirmation_ids':{lang:[d['id'] for d in docs] for lang,docs in fresh.items()}}
        usage_path = out/'dataset_usage.json'
        if usage_path.exists() and json.loads(usage_path.read_text(encoding='utf-8')) != usage:
            raise ValueError('Cannot resume follow-up: frozen evaluation data changed')
        dump(usage_path,usage)
        torch.set_num_threads(cfg['threads'])
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        checkpoints = [AutoModelForCausalLM.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision'],
            torch_dtype=torch.float32,attn_implementation='eager') for k in ['base','swedish']]
        tokenizers = [AutoTokenizer.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision']) for k in ['base','swedish']]
        compatibility = verify(*checkpoints,*tokenizers,{k:cfg[k+'_revision'] for k in ['base','swedish']})
        dump(out/'compatibility.json',compatibility)
        if not compatibility['compatible']:
            raise ValueError('Follow-up checkpoint compatibility failed')
        base,adapted = map(snapshot,checkpoints)
        delta = compute_delta(base,adapted)
        model = checkpoints[0]
        for fraction,expected in [(0,base),(1,adapted)]:
            patch,_ = make_balanced_patch(base,delta,fraction)
            apply_patch(model,base,patch)
            for name,p in model.named_parameters():
                torch.testing.assert_close(p.cpu(),expected[name],rtol=1e-5,atol=1e-7)
        del patch,adapted,checkpoints
        gc.collect()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        dump(out/'batch_equivalence.json',verify_equivalence(model,tokenizers[0],fresh['sv'][:2]+fresh['en'][:2],
                                                           cfg['context_length'],cfg['inference_batch_size']))
        schedules = {
            'diagnostic': [('balanced',f,0) for f in cfg['fractions']],
            'confirmation': [('base',0.,0),('adapted',1.,0)]+[(m,f,s) for f in cfg['fractions']
                for m in plan['methods'] for s in (cfg['seeds'] if m == 'random' else [0])]}
        for stage,conditions in schedules.items():
            directory = out/stage
            directory.mkdir(exist_ok=True)
            progress_path = directory/'condition_progress.json'
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding='utf-8'))
                rows,details,completed = progress['rows'],progress['details'],progress['completed']
            elif stage == 'diagnostic':
                rows = pd.read_csv(root/plan['original_results']/'experiment_results.csv').to_dict('records')
                details = json.loads((root/plan['original_results']/'document_metrics.json').read_text(encoding='utf-8'))
                completed = []
            else:
                rows,details,completed = [],{},[]
            documents = observed if stage == 'diagnostic' else fresh
            for method,fraction,seed in conditions:
                key = f'{method}:{fraction}:{seed}'
                if key in completed:
                    continue
                dump(out/'status.json',{'status':'running','stage':stage,'condition':key,
                    'completed_new_conditions':len(completed),'stage_conditions':len(conditions),'pid':os.getpid()})
                print(f'{stage}: {key}',flush=True)
                if method == 'balanced':
                    patch,info = make_balanced_patch(base,delta,fraction)
                    dump(directory/f'group_budgets_{fraction}.json',info.pop('group_budgets'))
                else:
                    patch,info = make_patch(base,delta,fraction,'magnitude' if method in ['base','adapted'] else method,seed)
                apply_patch(model,base,patch)
                del patch
                details[key] = {}
                for lang in ['sv','en']:
                    metric = evaluate_batched(model,tokenizers[0],documents[lang],cfg['context_length'],cfg['inference_batch_size'])
                    details[key][f'{lang}:test'] = metric
                    rows.append(dict(method=method,fraction=fraction,seed=seed,language=lang,split='test',**info,
                        **{k:v for k,v in metric.items() if k != 'per_document'}))
                completed.append(key)
                dump(directory/'document_metrics.json',details)
                pd.DataFrame(rows).to_csv(directory/'experiment_results.csv',index=False)
                dump(progress_path,dict(rows=rows,details=details,completed=completed))
            analyze_results(directory,plan['bootstrap_samples'])
            # Check that every method really scored the identical document targets.
            for lang,docs in documents.items():
                reference = details['base:0.0:0'][f'{lang}:test']['per_document']
                assert [d['id'] for d in reference] == [d['id'] for d in docs]
                for condition in details.values():
                    candidate = condition[f'{lang}:test']['per_document']
                    assert [(d['id'],d['tokens']) for d in candidate] == [(d['id'],d['tokens']) for d in reference]
            dump(directory/'audit.json',{'passed':True,'same_ids_and_token_counts_for_all_conditions':True,
                'documents_per_language':len(documents['sv']),'new_conditions':len(conditions)})
            dump(directory/'status.json',{'status':'complete','new_conditions':len(conditions)})
        for name,expected in identity['original_artifacts'].items():
            assert digest((root/plan['original_results']/name).read_bytes()) == expected, f'Original artifact changed: {name}'
        dump(out/'environment.json',{'device':device,'seconds':time.time()-start,
            'python':sys.version,'torch':torch.__version__,'original_quantitative_artifacts_unchanged':True})
        export_results(root)
        dump(out/'status.json',{'status':'complete','diagnostic_new_conditions':2,'confirmation_conditions':14,
            'finished_at':timestamp(),'results':'results/follow_up'})
        print('Follow-up finished: results/follow_up',flush=True)
    except Exception as exc:
        dump(out/'status.json',{'status':'failed','error':repr(exc)})
        raise


def run_dense_follow_up():
    """Select before collecting confirmation; preserve both previous studies."""
    from .datasets import collect, USER_AGENT, timestamp
    from .patches import scaled_dense_patch
    from .evaluation import select_dense_alpha, paired_bootstrap
    from .reporting import export_results
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/experiment.yaml').read_text())
    plan = yaml.safe_load((root/'configs/dense_follow_up.yaml').read_text())
    out = root/plan['results_directory']
    out.mkdir(parents=True,exist_ok=True)
    started = time.time()
    dump(out/'process.json',{'pid':os.getpid(),'started_at':timestamp()})
    def status(**fields):
        dump(out/'status.json',dict(pid=os.getpid(),**fields))
    try:
        status(status='validating')
        audit()
        unit = subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,capture_output=True,text=True)
        (out/'unit_checks.txt').write_text(unit.stdout+unit.stderr,encoding='utf-8')
        unit.check_returncode()
        artifacts = [p for directory in ['results/wikipedia/full','results/follow_up']
                     for p in (root/directory).rglob('*') if p.is_file()
                     and p.name not in ['audit.json','pipeline.lock']]
        identity = {'config':cfg,'plan':plan,
            'plan_sha256':digest((root/'reports/dense_follow_up_plan.md').read_bytes()),
            'source':{p.name:digest(p.read_bytes()) for p in (root/'src').glob('*.py')},
            'development_sha256':{lang:digest((root/f'data/sample_{lang}.jsonl').read_bytes()) for lang in ['sv','en']},
            'prior_artifacts':{str(p.relative_to(root)):digest(p.read_bytes()) for p in artifacts},
            'excluded_manifests':{d:digest((root/d/'dataset_manifest.json').read_bytes()) for d in plan['exclude_directories']}}
        design_path = out/'design.json'
        if design_path.exists():
            if json.loads(design_path.read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError('Dense follow-up design or evidence changed; refusing mixed results')
        else:
            dump(design_path,{'frozen_before_selection_at':timestamp(),'identity':identity})
        torch.set_num_threads(cfg['threads'])
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        checkpoints = [AutoModelForCausalLM.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision'],
            torch_dtype=torch.float32,attn_implementation='eager') for k in ['base','swedish']]
        tokenizers = [AutoTokenizer.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision']) for k in ['base','swedish']]
        compatibility = verify(*checkpoints,*tokenizers,{k:cfg[k+'_revision'] for k in ['base','swedish']})
        dump(out/'compatibility.json',compatibility)
        if not compatibility['compatible']:
            raise ValueError('Checkpoint compatibility failed')
        base,adapted = map(snapshot,checkpoints)
        delta = compute_delta(base,adapted)
        model = checkpoints[0]
        for alpha,expected in [(0,base),(1,adapted)]:
            patch = scaled_dense_patch(delta,alpha)
            apply_patch(model,base,patch)
            for name,p in model.named_parameters():
                torch.testing.assert_close(p.cpu(),expected[name],rtol=1e-5,atol=1e-7)
        del patch,adapted,checkpoints
        gc.collect()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        development = {lang:[json.loads(line) for line in (root/f'data/sample_{lang}.jsonl').read_text(encoding='utf-8').splitlines()]
                       for lang in ['sv','en']}
        assert all(len(docs) == 11 for docs in development.values())
        dump(out/'batch_equivalence.json',verify_equivalence(model,tokenizers[0],development['sv'][:2]+development['en'][:2],
            cfg['context_length'],cfg['inference_batch_size']))
        def score(documents):
            return {f'{lang}:test':evaluate_batched(model,tokenizers[0],documents[lang],cfg['context_length'],
                    cfg['inference_batch_size']) for lang in ['sv','en']}
        dev_path = out/'development_metrics.json'
        dev = json.loads(dev_path.read_text(encoding='utf-8')) if dev_path.exists() else {}
        for alpha in plan['alphas']:
            key = str(float(alpha))
            if key in dev:
                continue
            status(status='running',stage='development',alpha=alpha,completed=len(dev),conditions=len(plan['alphas']))
            print(f'development alpha={alpha}',flush=True)
            patch = scaled_dense_patch(delta,alpha)
            apply_patch(model,base,patch)
            del patch
            dev[key] = score(development)
            dump(dev_path,dev)
        selection = dict(select_dense_alpha(dev),development_sha256=identity['development_sha256'],
                         development_metrics_sha256=digest(dev_path.read_bytes()))
        selection_path = out/'selection.json'
        if selection_path.exists():
            previous = json.loads(selection_path.read_text(encoding='utf-8'))
            assert previous['selection'] == selection, 'Frozen selection changed'
        else:
            dump(selection_path,{'selected_before_collection_at':timestamp(),'selection':selection})
        print(f"Fixed selected alpha={selection['alpha']}; collecting confirmation now",flush=True)
        status(status='collecting',selected_alpha=selection['alpha'])
        confirmation_dir = root/plan['confirmation_directory']
        exclusions = [root/d for d in plan['exclude_directories']]
        collect(confirmation_dir,plan['confirmation_documents_per_language'],USER_AGENT,5000,exclusions)
        _,fresh = verify_frozen(confirmation_dir)
        old = [doc for directory in exclusions for docs in verify_frozen(directory)[1].values() for doc in docs]
        new = [doc for docs in fresh.values() for doc in docs]
        assert not ({(d['language'],d['article_id']) for d in old} & {(d['language'],d['article_id']) for d in new})
        for field in ['wikidata_id','content_sha256']:
            assert not ({d[field] for d in old if d.get(field)} & {d[field] for d in new if d.get(field)})
        usage = {'manifest_sha256':digest((confirmation_dir/'dataset_manifest.json').read_bytes()),
                 'exclusions':identity['excluded_manifests'],'disjoint_pages_topics_hashes':True,
                 'selection_sha256':digest(selection_path.read_bytes())}
        usage_path = out/'dataset_usage.json'
        if usage_path.exists():
            assert json.loads(usage_path.read_text(encoding='utf-8')) == usage, 'Confirmation data changed'
        dump(usage_path,usage)
        directory = out/'confirmation'
        directory.mkdir(exist_ok=True)
        progress_path = directory/'condition_progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {'rows':[],'details':{},'completed':[]}
        rows,details,completed = progress['rows'],progress['details'],progress['completed']
        conditions = [('base',0.),('adapted',1.),('dense',1.),('balanced',.3),('magnitude',.3)]
        total = sum(t.numel() for t in delta.values())
        for method,fraction in conditions:
            key = f'{method}:{fraction}:0'
            if key in completed:
                continue
            status(status='running',stage='confirmation',condition=key,completed=len(completed),conditions=len(conditions),selected_alpha=selection['alpha'])
            print(f'confirmation {key}',flush=True)
            alpha = selection['alpha'] if method == 'dense' else fraction
            if method in ['base','adapted','dense']:
                patch = scaled_dense_patch(delta,alpha)
                info = {'selected_parameters':0 if method == 'base' else total,'total_parameters':total,
                        'retained_fraction':fraction,'alpha':alpha}
            elif method == 'balanced':
                patch,info = make_balanced_patch(base,delta,fraction)
                dump(directory/'balanced_group_budgets.json',info.pop('group_budgets'))
            else:
                patch,info = make_patch(base,delta,fraction,method)
            endpoint = f"{'base' if alpha == 0 else 'adapted'}:{0.0 if alpha == 0 else 1.0}:0"
            if method == 'dense' and alpha in [0,1]:
                details[key] = details[endpoint]
            else:
                apply_patch(model,base,patch)
                details[key] = score(fresh)
            del patch
            for lang in ['sv','en']:
                rows.append(dict(method=method,fraction=fraction,seed=0,language=lang,split='test',**info,
                    **{k:v for k,v in details[key][f'{lang}:test'].items() if k != 'per_document'}))
            completed.append(key)
            dump(directory/'document_metrics.json',details)
            pd.DataFrame(rows).to_csv(directory/'experiment_results.csv',index=False)
            dump(progress_path,dict(rows=rows,details=details,completed=completed))
        analyze_results(directory,plan['bootstrap_samples'])
        joint_draws = np.zeros(plan['bootstrap_samples'])
        differences = []
        language_comparisons = {}
        for lang in ['sv','en']:
            reference = details['balanced:0.3:0'][f'{lang}:test']['per_document']
            candidate = details['dense:1.0:0'][f'{lang}:test']['per_document']
            language_comparisons[lang] = paired_bootstrap(reference,candidate,plan['bootstrap_samples'],plan['bootstrap_seed'])
            rng = np.random.default_rng(np.random.SeedSequence([plan['bootstrap_seed'],0 if lang == 'sv' else 1]))
            draws = rng.integers(0,len(reference),(plan['bootstrap_samples'],len(reference)))
            losses = np.array([c['loss_sum']-r['loss_sum'] for r,c in zip(reference,candidate)])
            tokens = np.array([r['tokens'] for r in reference])
            joint_draws += .5*losses[draws].sum(1)/tokens[draws].sum(1)
            differences.append(losses.sum()/tokens.sum())
            for condition in details.values():
                docs = condition[f'{lang}:test']['per_document']
                assert [d['id'] for d in docs] == [d['id'] for d in fresh[lang]]
                assert [(d['id'],d['tokens']) for d in docs] == [(d['id'],d['tokens']) for d in reference]
        lo,hi = np.quantile(joint_draws,[.025,.975])
        dump(out/'primary_comparison.json',{'candidate':'selected dense','reference':'balanced 30%',
            'delta_mean_language_nll':float(np.mean(differences)),'ci_low':float(lo),'ci_high':float(hi),
            'languages':language_comparisons,'samples':plan['bootstrap_samples'],'seed':plan['bootstrap_seed'],
            'resampling':'Independent paired document draws within each language, equal language weighting'})
        for filename,expected in identity['prior_artifacts'].items():
            assert digest((root/filename).read_bytes()) == expected, f'Prior artifact changed: {filename}'
        dump(out/'audit.json',{'passed':True,'prior_artifacts_unchanged':True,'selection_frozen_before_collection':True,
            'same_document_ids_and_token_counts':True,'disjoint_from_both_previous_corpora':True,
            'documents_per_language':plan['confirmation_documents_per_language'],'conditions':len(conditions)})
        dump(out/'environment.json',{'device':device,'seconds':time.time()-started,'python':sys.version,
            'torch':torch.__version__,'platform':platform.platform()})
        export_results(root)
        status(status='complete',selected_alpha=selection['alpha'],confirmation_conditions=len(conditions),
               finished_at=timestamp(),results='results/dense_follow_up')
        print('Complete: results/dense_follow_up',flush=True)
    except Exception as exc:
        status(status='failed',error=repr(exc))
        raise


def scientific_audit(check_weights=False):
    """Recompute evidence without overwriting any recorded inference measurement."""
    from .datasets import timestamp
    from .evaluation import paired_bootstrap, select_dense_alpha
    import tempfile
    root = Path(__file__).resolve().parents[1]
    output = root/'results/evaluator_audit'
    output.mkdir(exist_ok=True)
    evidence = {'at':timestamp(),'inference_rerun':False,
        'scope':'Technical consistency only; not certification of causality, training independence or submission completeness',
        'checks':{},'limitations':[
        'No independent clean-environment installation or public submission verified.',
        'Corpus disjointness does not establish training-data independence.',
        'Bootstrap uncertainty conditions on the fixed checkpoints, masks and sampled domain.',
        'Swedish card metadata names the Instruct parent; prose and paper describe original-model CPT. Ancestry remains unverified.']}
    checks = evidence['checks']
    audit()
    legacy = json.loads((root/'results/full/document_metrics.json').read_text(encoding='utf-8'))
    legacy_csv = pd.read_csv(root/'results/full/experiment_results.csv')
    assert len(legacy) == 24 and len(legacy_csv) == 96
    for key,condition in legacy.items():
        method,fraction,seed = key.split(':')
        for metric_key,metric in condition.items():
            lang,split = metric_key.split(':')
            source = [json.loads(line) for line in (root/f'data/sample_{lang}.jsonl').read_text(encoding='utf-8').splitlines()]
            docs = metric['per_document']
            assert [d['id'] for d in docs] == [d['id'] for d in source if d['split'] == split]
            assert len(docs) == (8 if split == 'test' else 3)
            assert metric['tokens'] == sum(d['tokens'] for d in docs)
            assert abs(metric['nll']-sum(d['loss_sum'] for d in docs)/metric['tokens']) < 1e-12
            row = legacy_csv[(legacy_csv.method == method)&(legacy_csv.fraction == float(fraction))&
                (legacy_csv.seed == int(seed))&(legacy_csv.language == lang)&(legacy_csv.split == split)]
            assert len(row) == 1 and abs(row.iloc[0].nll-metric['nll']) < 1e-12
    checks['original_synthetic_study'] = {'conditions':24,'rows':96,'per_document_sums_and_original_split_ids_match':True}
    corpora = {}
    for name,expected in [('data',150),('data/follow_up',50),('data/dense_follow_up',50)]:
        manifest,docs = verify_frozen(root/name)
        assert manifest['requested_per_language'] == expected
        corpora[name] = docs
        accepted = {}
        for lang in ['sv','en']:
            entry = manifest['datasets'][lang]
            decisions = json.loads((root/name/entry['audit_file']).read_text(encoding='utf-8'))
            ids = {d['article_id'] for d in docs[lang]}
            assert {d['article_id'] for d in decisions if d['decision'] == 'accepted'} == ids
            accepted[lang] = len(ids)
        checks[name] = {'frozen_hashes_provenance_licenses_and_audits':True,'documents':accepted,
                        'manifest_sha256':digest((root/name/'dataset_manifest.json').read_bytes())}
    for a,b in [('data','data/follow_up'),('data','data/dense_follow_up'),('data/follow_up','data/dense_follow_up')]:
        left,right = [d for docs in corpora[a].values() for d in docs],[d for docs in corpora[b].values() for d in docs]
        assert not ({(d['language'],d['article_id']) for d in left} & {(d['language'],d['article_id']) for d in right})
        for key in ['content_sha256','wikidata_id']:
            assert not ({d[key] for d in left if d.get(key)} & {d[key] for d in right if d.get(key)})
    checks['corpus_disjointness'] = 'All three corpora disjoint by recorded page, topic and text hash'
    stages = [('results/wikipedia/full','data',24),('results/follow_up/diagnostic','data',26),
              ('results/follow_up/confirmation','data/follow_up',14),
              ('results/dense_follow_up/confirmation','data/dense_follow_up',5)]
    all_details = {}
    examples = []
    for relative,corpus,expected in stages:
        directory = root/relative
        details = json.loads((directory/'document_metrics.json').read_text(encoding='utf-8'))
        all_details[relative] = details
        frame = pd.read_csv(directory/'experiment_results.csv')
        assert len(details) == expected
        for key,condition in details.items():
            method,fraction,seed = key.split(':')
            for lang in ['sv','en']:
                metric = condition[f'{lang}:test']
                docs = metric['per_document']
                assert [d['id'] for d in docs] == [d['id'] for d in corpora[corpus][lang]]
                reference = details['base:0.0:0'][f'{lang}:test']['per_document']
                assert [(d['id'],d['tokens'],d['chunks']) for d in docs] == [(d['id'],d['tokens'],d['chunks']) for d in reference]
                assert all(d['tokens'] > 0 and np.isfinite(d['loss_sum']) and d['loss_sum'] >= 0 for d in docs)
                assert metric['tokens'] == sum(d['tokens'] for d in docs)
                assert metric['documents'] == len(docs)
                assert metric['examples'] == sum(d['chunks'] for d in docs)
                assert abs(metric['nll']-sum(d['loss_sum'] for d in docs)/metric['tokens']) < 1e-12
                row = frame[(frame.method == method) & (frame.fraction == float(fraction)) &
                            (frame.seed == int(seed)) & (frame.language == lang) & (frame.split == 'test')]
                assert len(row) == 1 and abs(row.iloc[0].nll-metric['nll']) < 1e-12
        with tempfile.TemporaryDirectory(dir=root/'.cache') as temporary:
            target = Path(temporary)
            for name in ['document_metrics.json','experiment_results.csv']:
                shutil.copyfile(directory/name,target/name)
            analyze_results(target,2000)
            for name in ['method_summary.csv','paired_comparisons.csv','paired_vs_base_per_seed.csv']:
                pd.testing.assert_frame_equal(pd.read_csv(directory/name),pd.read_csv(target/name),
                                              check_exact=False,rtol=1e-12,atol=1e-12)
        checks[relative] = {'conditions':expected,'document_losses_match_csv':True,
            'identical_ids_targets_and_chunks':True,'all_summary_and_paired_intervals_recomputed':True}
        if not relative.endswith('diagnostic'):
            target = ('layer:0.3:0' if relative.endswith('full') else
                      'balanced:0.3:0' if corpus == 'data/follow_up' else 'dense:1.0:0')
            for lang in ['sv','en']:
                for index,doc in enumerate(corpora[corpus][lang][:2]):
                    row = {'corpus':corpus,'language':lang,'id':doc['id'],'source_url':doc['source_url'],
                           'revision_url':doc['revision_url'],'comparison_condition':target}
                    for label,key in [('base','base:0.0:0'),('adapted','adapted:1.0:0'),('candidate',target)]:
                        metric = details[key][f'{lang}:test']['per_document'][index]
                        row[label+'_nll'] = metric['loss_sum']/metric['tokens']
                        row['tokens'] = metric['tokens']
                    examples.append(row)
    pd.DataFrame(examples).to_csv(output/'document_examples.csv',index=False)
    dense = root/'results/dense_follow_up'
    selection = json.loads((dense/'selection.json').read_text(encoding='utf-8'))
    development = json.loads((dense/'development_metrics.json').read_text(encoding='utf-8'))
    actual = select_dense_alpha(development)
    assert all(selection['selection'][k] == v for k,v in actual.items())
    assert selection['selection']['development_metrics_sha256'] == digest((dense/'development_metrics.json').read_bytes())
    manifest = json.loads((root/'data/dense_follow_up/dataset_manifest.json').read_text(encoding='utf-8'))
    assert selection['selected_before_collection_at'] < manifest['started_at']
    dense_details = all_details['results/dense_follow_up/confirmation']
    if actual['alpha'] in [0,1]:
        endpoint = 'base:0.0:0' if actual['alpha'] == 0 else 'adapted:1.0:0'
        assert dense_details['dense:1.0:0'] == dense_details[endpoint]
    primary = json.loads((dense/'primary_comparison.json').read_text(encoding='utf-8'))
    draws_mean = np.zeros(2000)
    for index,lang in enumerate(['sv','en']):
        ref = dense_details['balanced:0.3:0'][f'{lang}:test']['per_document']
        candidate = dense_details['dense:1.0:0'][f'{lang}:test']['per_document']
        assert primary['languages'][lang] == paired_bootstrap(ref,candidate)
        rng = np.random.default_rng(np.random.SeedSequence([2026,index]))
        draws = rng.integers(0,len(ref),(2000,len(ref)))
        loss = np.array([c['loss_sum']-r['loss_sum'] for r,c in zip(ref,candidate)])
        tokens = np.array([r['tokens'] for r in ref])
        draws_mean += .5*loss[draws].sum(1)/tokens[draws].sum(1)
    np.testing.assert_allclose([primary['ci_low'],primary['ci_high']],np.quantile(draws_mean,[.025,.975]),rtol=0,atol=1e-12)
    checks['dense_selection'] = {'selected_alpha':actual['alpha'],'selected_before_collection':True,
                               'endpoint_reuse_valid':True,'primary_bilingual_interval_recomputed':True}
    from .reporting import validate_notes
    checks['author_notes'] = validate_notes(root)
    for report in ['results/follow_up/design.json','results/dense_follow_up/design.json']:
        identity = json.loads((root/report).read_text(encoding='utf-8'))['identity']
        artifacts = identity.get('original_artifacts')
        if artifacts:
            for name,expected in artifacts.items():
                assert digest((root/'results/wikipedia/full'/name).read_bytes()) == expected
        else:
            for name,expected in identity['prior_artifacts'].items():
                assert digest((root/name).read_bytes()) == expected
    checks['historical_inference_artifacts_unchanged'] = True
    sources = output/'external_sources.json'
    if sources.exists():
        external = json.loads(sources.read_text(encoding='utf-8'))
        declared = external['swedish']['base_model']
        configured = yaml.safe_load((root/'configs/experiment.yaml').read_text())['base_model']
        checks['checkpoint_ancestry'] = {'configured_base':configured,'declared_card_parent':declared,
            'metadata_matches_configured_base':configured in (declared if isinstance(declared,list) else [declared]),
            'status':'unresolved metadata conflict; structural compatibility alone cannot resolve lineage'}
    source_root = Path(os.environ.get('PROJECT_SOURCE_ROOT',root))
    git = subprocess.run(['git','-c',f'safe.directory={source_root.as_posix()}','ls-files'],cwd=source_root,capture_output=True,text=True)
    checks['git_tracked_files'] = git.stdout.splitlines()
    if check_weights:
        checks['real_checkpoint_masks'] = audit_real_masks(root,corpora)
    evidence['passed'] = True
    dump(output/('weight_verification.json' if check_weights else 'verification.json'),evidence)
    print(json.dumps(evidence,indent=2))


def audit_real_masks(root,corpora):
    """Rebuild masks, not model predictions, using pinned cached weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = yaml.safe_load((root/'configs/experiment.yaml').read_text())
    torch.set_num_threads(cfg['threads'])
    models = [AutoModelForCausalLM.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision'],
        torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True) for k in ['base','swedish']]
    tokenizers = [AutoTokenizer.from_pretrained(cfg[k+'_model'],revision=cfg[k+'_revision'],local_files_only=True) for k in ['base','swedish']]
    assert verify(*models,*tokenizers,{k:cfg[k+'_revision'] for k in ['base','swedish']})['compatible']
    base,adapted = map(snapshot,models)
    delta = compute_delta(base,adapted)
    del adapted,models
    gc.collect()
    total = sum(t.numel() for t in delta.values())
    saved = pd.read_csv(root/'results/wikipedia/full/experiment_results.csv')
    records = []
    for fraction in cfg['fractions']:
        for method in ['random','dare','magnitude','layer','layer_random']:
            for seed in cfg['seeds'] if method in ['random','dare','layer_random'] else [0]:
                patch,info = make_patch(base,delta,fraction,method,seed)
                assert info['selected_parameters'] == int(np.floor(total*fraction))
                scale = info['rescale']
                for name,value in patch.items():
                    nonzero = value != 0
                    expected = delta[name]*scale
                    assert torch.equal(value[nonzero],expected[nonzero])
                    if method == 'magnitude' and nonzero.any() and (~nonzero).any():
                        assert float(delta[name][nonzero].abs().min()) >= float(delta[name][~nonzero].abs().max())
                row = saved[(saved.method == method)&(saved.fraction == fraction)&(saved.seed == seed)].iloc[0]
                for name in ['selected_parameters','modified_parameters','total_parameters','retained_fraction','rescale','delta_l2']:
                    np.testing.assert_allclose(info[name],row[name],rtol=1e-7,atol=1e-7)
                if method == 'dare':
                    # Exact same mask at each seed; only the inverse retention scale differs.
                    random,_ = make_patch(base,delta,fraction,'random',seed)
                    assert all(torch.equal(patch[n],random[n]*scale) for n in delta)
                    del random
                del patch
                records.append({'method':method,'fraction':fraction,'seed':seed,'passed':True,'rescale':scale})
                print(f'Mask checked: {method} {fraction} {seed}',flush=True)
        patch,info = make_balanced_patch(base,delta,fraction)
        assert info['selected_parameters'] == int(np.floor(total*fraction))
        previous = json.loads((root/f'results/follow_up/confirmation/group_budgets_{fraction}.json').read_text())
        assert info['group_budgets'] == previous
        records.append({'method':'balanced','fraction':fraction,'seed':0,'passed':True})
        del patch
    token_checks = {}
    from math import ceil
    for corpus,documents in corpora.items():
        relative = {'data':'results/wikipedia/full','data/follow_up':'results/follow_up/confirmation',
                    'data/dense_follow_up':'results/dense_follow_up/confirmation'}[corpus]
        details = json.loads((root/relative/'document_metrics.json').read_text(encoding='utf-8'))['base:0.0:0']
        for lang,docs in documents.items():
            for doc,row in zip(docs,details[f'{lang}:test']['per_document']):
                targets = len(tokenizers[0].encode(doc['text'],add_special_tokens=False))-1
                assert row['tokens'] == targets and row['chunks'] == ceil(targets/(cfg['context_length']-1))
            token_checks[f'{corpus}:{lang}'] = len(docs)
    return {'total_unique_parameters':total,'masks':records,'all_recorded_counts_norms_and_scales_match':True,
            'random_dare_identical_masks_verified':True,'tokenizer_target_and_chunk_counts_verified':token_checks,
            'likelihood_inference_repeated':False}


if __name__ == '__main__':
    main()
