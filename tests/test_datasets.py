import json
from pathlib import Path
import pytest
from src.datasets import clean_prose, digest, verify_frozen, RULES, LICENSE, API


def paragraph(label, n=30):
    return '<p>'+(' '.join([label]*n))+'.</p>'


def test_evidence_bundle_round_trip_preserves_exact_bytes(tmp_path):
    from run_project import write_bundle,unpack_bundle
    source = tmp_path/'source'
    (source/'data').mkdir(parents=True)
    content = 'Svensk text: åäö\r\n'.encode('utf-8')
    (source/'data/text.jsonl').write_bytes(content)
    archive = tmp_path/'evidence.zip'
    write_bundle(source,archive,['data/text.jsonl'])
    unpack_bundle(archive,tmp_path/'restored')
    assert (tmp_path/'restored/data/text.jsonl').read_bytes() == content


def test_evidence_bundle_rejects_traversal_and_modified_bytes(tmp_path):
    import zipfile
    from run_project import read_bundle
    archive = tmp_path/'unsafe.zip'
    for name,payload,expected in [('../escape',b'x',digest(b'x')),('data/text',b'changed',digest(b'original'))]:
        with zipfile.ZipFile(archive,'w') as handle:
            handle.writestr(name,payload)
            handle.writestr('bundle_manifest.json',json.dumps({name:expected}))
        with pytest.raises(ValueError):
            read_bundle(archive)


def test_result_publication_never_overwrites_author_reports(tmp_path,monkeypatch):
    import run_project as launcher
    project,work = tmp_path/'project',tmp_path/'work'
    for directory in ['reports','data','results']:
        (project/directory).mkdir(parents=True)
        (work/directory).mkdir(parents=True)
    originals = {'research_note.md':b'Author text','research_note.pdf':b'Author PDF bytes'}
    for name,data in originals.items():
        (project/'reports'/name).write_bytes(data)
        (work/'reports'/name).write_bytes(b'Other text that must never replace the author')
    (work/'reports/old.pdf').write_bytes(b'Historical PDF')
    (work/'data/frozen.json').write_bytes(b'{}')
    monkeypatch.setattr(launcher,'PROJECT_ROOT',project)
    monkeypatch.setattr(launcher,'ROOT',work)
    launcher.publish_workspace()
    assert {p.name:p.read_bytes() for p in (project/'reports').iterdir()} == originals
    assert 'reports/old.pdf' in launcher.read_bundle(project/'results/evidence.zip')


def test_table_export_does_not_edit_notes_or_readme(tmp_path):
    from src.reporting import export_results
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/research_note.md').write_text('My note')
    (tmp_path/'README.md').write_text('Brief README')
    main = tmp_path/'results/wikipedia/full'
    (main/'figures').mkdir(parents=True)
    (main/'figures/wikipedia_nll.png').write_bytes(b'existing figure')
    (main/'method_summary.csv').write_text('method,fraction,language,nll\nbase,0,sv,4.1\nbase,0,en,2.9\n')
    for name in ['experiment_results.csv','paired_comparisons.csv']:
        (main/name).write_text('recorded\n')
    export_results(tmp_path)
    assert (tmp_path/'reports/research_note.md').read_text() == 'My note'
    assert (tmp_path/'README.md').read_text() == 'Brief README'
    assert sorted(p.name for p in (tmp_path/'reports').iterdir()) == ['research_note.md']
    assert (tmp_path/'results/method_summary.csv').read_bytes() == (main/'method_summary.csv').read_bytes()


def test_prose_excludes_markup_reference_and_uses_article_body():
    html = '<div class="mw-parser-output"><table>'+paragraph('table',200)+'</table>'
    html += '<div class="hatnote">'+paragraph('notice',200)+'</div>'
    html += paragraph('introduction')+'<h2>History</h2>'+paragraph('history')*4
    html += '<h2>References</h2>'+paragraph('reference',200)+'</div>'
    text,stats = clean_prose(html)
    assert 100 <= len(text.split()) <= 200
    assert 'table' not in text and 'notice' not in text and 'reference' not in text
    assert 'history' in text and stats['cleaned_article_prose_words'] == 150
    assert text.endswith('.')


def test_insufficient_prose_and_list_only_rejected():
    with pytest.raises(ValueError,match='insufficient'):
        clean_prose('<ul><li>'+('word '*300)+'</li></ul>')
    with pytest.raises(ValueError):
        clean_prose(paragraph('short',20))


def frozen_fixture(tmp_path):
    manifest = {'status':'complete','requested_per_language':1,'datasets':{}}
    for lang in ['sv','en']:
        text = ' '.join([lang]*120)+'.'
        row = dict(id=lang,language=lang,split='test',article_id=1,revision_id=2,text=text,
                   content_sha256=digest(text),source_url='https://example.org',revision_url='https://example.org/2',
                   history_url='https://example.org/history',retrieved_at='2026-11-19T00:00:00+00:00',
                   license=LICENSE,attribution='contributors',modifications=RULES['modifications'])
        path = tmp_path/f'eval_{lang}.jsonl'
        path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        manifest['datasets'][lang] = dict(file=path.name,sha256=digest(path.read_bytes()),documents=1,ordered_ids=[lang])
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')


def test_frozen_hash_and_count_are_enforced(tmp_path):
    frozen_fixture(tmp_path)
    assert len(verify_frozen(tmp_path)[1]['sv']) == 1
    with (tmp_path/'eval_sv.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_frozen(tmp_path)


def test_api_honors_retry_after_and_cache(tmp_path,monkeypatch):
    from src import datasets as module
    waits = []
    monkeypatch.setattr(module.time,'sleep',lambda seconds:waits.append(seconds))
    class Reply:
        headers = {'Retry-After':'7'}
        url = 'https://sv.wikipedia.org/w/api.php'
        def raise_for_status(self):
            pass
        def json(self):
            return {'error':{'code':'maxlag'}} if calls[0] == 1 else {'query':{}}
    calls = [0]
    api = API(tmp_path,'test agent')
    def get(*args,**kwargs):
        calls[0] += 1
        return Reply()
    monkeypatch.setattr(api.session,'get',get)
    result = api.get('sv',{'action':'query'})
    assert calls[0] == 2 and 7 in waits and len(api.events) == 1
    assert api.get('sv',{'action':'query'}) == result
    assert calls[0] == 2


def test_transient_file_lock_retried_without_losing_state(tmp_path,monkeypatch):
    from src import datasets as module
    original = module.save
    calls = []
    def locked(path,value):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError('temporary sharing violation')
        original(path,value)
    monkeypatch.setattr(module,'save',locked)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    path = tmp_path/'progress.json'
    module.save_retry(path,{'completed':15})
    assert len(calls) == 3 and json.loads(path.read_text()) == {'completed':15}


def test_frozen_audit_and_license_tampering_rejected(tmp_path):
    frozen_fixture(tmp_path)
    manifest_path = tmp_path/'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    audit = tmp_path/'audit.json'
    audit.write_text('[]')
    manifest['datasets']['sv'].update(audit_file=audit.name,audit_sha256=digest(audit.read_bytes()))
    manifest_path.write_text(json.dumps(manifest))
    audit.write_text('[1]')
    with pytest.raises(ValueError,match='audit hash'):
        verify_frozen(tmp_path)
    audit.write_text('[]')
    path = tmp_path/'eval_sv.jsonl'
    row = json.loads(path.read_text())
    row['license'] = {'name':'unknown'}
    path.write_text(json.dumps(row)+'\n')
    manifest['datasets']['sv']['sha256'] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='license'):
        verify_frozen(tmp_path)
