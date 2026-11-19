"""One-command dataset validation, evaluation, result tables and plots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import shutil
import zipfile
from pathlib import PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
OUTPUT = ROOT/'results'/'wikipedia'/'full'


def write_bundle(source, target, files):
    """Keep exact evidence bytes in one deterministic, independently hashed ZIP."""
    source = source.resolve()
    members = {}
    for name in sorted(files):
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'Invalid bundle input: {name}')
        members[name] = path.read_bytes()
    manifest = {name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in {**members,'bundle_manifest.json':json.dumps(manifest,sort_keys=True,indent=2).encode()}.items():
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    # Validate the finished archive before replacing an existing bundle.
    read_bundle(temporary)
    temporary.replace(target)


def read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(archive.read('bundle_manifest.json'))
        if set(names) != set(manifest)|{'bundle_manifest.json'}:
            raise ValueError('Archive contents differ from its manifest')
        result = {}
        for name,expected in manifest.items():
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Archive hash mismatch: {name}')
            result[name] = data
        return result


def unpack_bundle(path, destination):
    destination = destination.resolve()
    members = read_bundle(path)
    for name,data in members.items():
        target = (destination/name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Archive extraction escapes working directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)


def prepare_workspace():
    global ROOT, OUTPUT
    ROOT = PROJECT_ROOT/'.cache/work'
    ROOT.mkdir(parents=True,exist_ok=True)
    bundles = [PROJECT_ROOT/'data/datasets.zip',PROJECT_ROOT/'results/evidence.zip']
    identity = {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundles}
    marker = ROOT/'bundle_identity.json'
    fresh = not marker.exists() or json.loads(marker.read_text()) != identity
    for bundle in bundles:
        if fresh:
            unpack_bundle(bundle,ROOT)
        else:
            read_bundle(bundle)
    marker.write_text(json.dumps(identity,indent=2))
    # Always run the editable source and tests, never an archived code snapshot.
    for directory in ['src','tests','configs']:
        for source in (PROJECT_ROOT/directory).glob('*'):
            if source.is_file():
                target = ROOT/directory/source.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source,target)
    for name in ['README.md','run_project.py','requirements.txt']:
        shutil.copyfile(PROJECT_ROOT/name,ROOT/name)
    (ROOT/'.cache').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    for source in (PROJECT_ROOT/'reports').glob('*'):
        if source.is_file():
            shutil.copyfile(source,ROOT/'reports'/source.name)
    OUTPUT = ROOT/'results/wikipedia/full'


def publish_workspace():
    # Raw measurements/designs retain their original paths inside the archives.
    data = [p.relative_to(ROOT).as_posix() for p in (ROOT/'data').rglob('*') if p.is_file()
            and '.collection_cache' not in p.parts and p.name not in ['collection_progress.json','collection_progress_lead_pilot.json']]
    results = [p.relative_to(ROOT).as_posix() for p in (ROOT/'results').rglob('*') if p.is_file() and p.name != 'pipeline.lock']
    # Retain historical reports/plans inside the evidence archive only.
    archived_reports = [p.relative_to(ROOT).as_posix() for p in (ROOT/'reports').rglob('*')
                        if p.is_file() and p.relative_to(ROOT).as_posix() not in
                        ['reports/research_note.md','reports/research_note.pdf','reports/research_note.docx']]
    write_bundle(ROOT,PROJECT_ROOT/'data/datasets.zip',data)
    write_bundle(ROOT,PROJECT_ROOT/'results/evidence.zip',results+archived_reports)
    visible = ['results/'+name for name in
               ['experiment_results.csv','method_summary.csv','paired_comparisons.csv','follow_up_summary.csv']]
    visible += [p.relative_to(ROOT).as_posix() for p in (ROOT/'results/figures').glob('*.png')
                if p.name != 'swedish_nll.png']
    for name in visible:
        source,target = ROOT/name,PROJECT_ROOT/name
        if source.is_file() and (not target.exists() or source.read_bytes() != target.read_bytes()):
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)


def run_module(module, *arguments):
    command = [sys.executable, '-m', module, *arguments]
    print('\n> '+' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)


def environment():
    env = os.environ.copy()
    # Support this workspace's existing local install as well as normal virtualenvs.
    local_dependencies = PROJECT_ROOT/'.deps'
    if local_dependencies.is_dir():
        env['PYTHONPATH'] = str(local_dependencies)+os.pathsep+env.get('PYTHONPATH', '')
    env.setdefault('HF_HOME', str(PROJECT_ROOT/'.cache'/'huggingface'))
    env['PROJECT_SOURCE_ROOT'] = str(PROJECT_ROOT)
    env.setdefault('PYTHONUTF8', '1')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted experiment; stop any existing experiment process first.')
    parser.add_argument('--results-only', dest='results_only', action='store_true',
                        help='Refresh tables and plots from completed measurements without editing research notes.')
    parser.add_argument('--follow-up',action='store_true',help='Run the separate balanced-layer follow-up and export its results')
    parser.add_argument('--dense-follow-up',action='store_true',help='Select scaled dense adaptation on development data and confirm on fresh texts')
    parser.add_argument('--audit-project',action='store_true',help='Run tests and verify all completed experiments without model inference')
    parser.add_argument('--check-weights',action='store_true',help='Also rebuild real-checkpoint masks during --audit-project')
    parser.add_argument('--unpack-only',action='store_true',help='Verify and unpack exact research artifacts into .cache/work')
    args = parser.parse_args()
    lock = PROJECT_ROOT/'.cache/pipeline.lock'
    lock.parent.mkdir(exist_ok=True)
    try:
        with lock.open('x') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        parser.exit(2,'Another pipeline holds .cache/pipeline.lock. Confirm it has stopped before removing a stale lock.\n')
    try:
        prepare_workspace()
        if args.unpack_only:
            print('Verified artifacts available at '+str(ROOT),flush=True)
            return
        workflow(args,parser)
        publish_workspace()
    finally:
        lock.unlink(missing_ok=True)


def workflow(args,parser):
    if args.audit_project:
        if args.results_only or args.follow_up or args.dense_follow_up or args.resume:
            parser.error('--audit-project is a separate action')
        run_module('pytest','-q','tests')
        run_module('src.run_experiment','--scientific-audit',*(['--check-weights'] if args.check_weights else []))
        return
    if args.check_weights:
        parser.error('--check-weights requires --audit-project')
    if args.follow_up or args.dense_follow_up:
        if args.results_only:
            parser.error('Follow-up experiments cannot be combined with --results-only')
        if args.follow_up and args.dense_follow_up:
            parser.error('Choose one follow-up experiment')
        name = 'dense_follow_up' if args.dense_follow_up else 'follow_up'
        flag = '--dense-follow-up' if args.dense_follow_up else '--follow-up'
        output = ROOT/'results'/name
        output.mkdir(parents=True,exist_ok=True)
        follow_lock = output/'pipeline.lock'
        try:
            with follow_lock.open('x') as handle:
                handle.write(str(os.getpid()))
        except FileExistsError:
            parser.exit(2,'A follow-up pipeline already holds the lock.\n')
        try:
            print(f'Local log: {ROOT}/results/{name}/run.log',flush=True)
            status = output/'status.json'
            if status.exists() and json.loads(status.read_text())['status'] == 'complete':
                run_module('src.run_experiment','--scientific-audit')
                run_module('src.reporting')
                print('Reused completed follow-up; no inference repeated.',flush=True)
                return
            with (output/'run.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','src.run_experiment',flag],cwd=ROOT,
                    env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
            run_module('src.reporting')
            print(f'Follow-up results complete: results/{name}',flush=True)
        finally:
            follow_lock.unlink(missing_ok=True)
        return
    status_path = OUTPUT/'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if status.get('status') == 'running' and not args.resume:
        parser.exit(2, 'An experiment is marked running. Let it finish before running this script.\n'
                       'If it was interrupted, stop its process before using --resume.\n')
    if args.results_only and status.get('status') != 'complete':
        parser.exit(2, 'Result export requires a completed full experiment.\n')
    # An exclusive launcher lock prevents two one-command pipelines from overlapping.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = OUTPUT/'pipeline.lock'
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        parser.exit(2, 'Another pipeline has a lock. If it crashed, remove results/wikipedia/full/pipeline.lock after confirming it has stopped.\n')
    try:
        run_module('src.datasets')
        if status.get('status') == 'complete' or args.results_only:
            run_module('pytest', '-q', 'tests')
            print('Reusing completed experiment; the audit verifies data and source identity.', flush=True)
        else:
            # run_experiment runs the entire test suite before model evaluation.
            run_module('src.run_experiment', '--config', 'configs/experiment.yaml', '--full', '--resume')
        run_module('src.run_experiment', '--audit')
        run_module('src.reporting')
        print('\nFinished. Tables and plots: '+str(PROJECT_ROOT/'results'), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Stop any remaining child process before resuming.')
