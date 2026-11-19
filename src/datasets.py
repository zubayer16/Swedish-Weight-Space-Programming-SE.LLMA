"""Rate-limited Wikipedia collection, attribution and frozen corpus verification."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


USER_AGENT = ('SELLMAWeightSpaceEvaluation/1.0 '
              '(https://github.com/zubayer16/Swedish-Weight-Space-Programming-SE.LLMA; research corpus collection)')


LICENSE = {'name': 'Creative Commons Attribution-ShareAlike 4.0 International',
           'spdx': 'CC-BY-SA-4.0', 'url': 'https://creativecommons.org/licenses/by-sa/4.0/',
           'terms_url': 'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content'}


RULES = {'version': 2, 'namespace': 0, 'candidate_min_bytes': 5000,
         'candidate_batch_size': 20, 'min_words': 100, 'target_words': 120, 'max_words': 200,
         'sampling': 'API random generator; first qualifying pages in returned index order; no model-based selection',
         'cleaning': 'Revision HTML prose p elements in article order until bibliography/references/external-links heading; exclude tables/lists/notices/media/references; NFC and whitespace normalization; sentence-boundary prefix of 100-200 whitespace words, first boundary >=120 preferred',
         'paragraph_filter': 'At least 20 whitespace words, >=60% of nonspace characters alphabetic, and sentence punctuation',
         'authorship': 'Exclude marked bot-generated/AI-generated articles; community-authored provenance, not a guarantee of exclusively human editing',
         'exclusions': 'Redirect/disambiguation/list pages, insufficient prose, automated-generation markers, copyright problem markers, duplicate page/content/Wikidata item',
         'truncation': 'Explicit prose excerpt at preparation only; scorer evaluates every frozen token except document first token',
         'modifications': 'HTML/markup, citations, notices and nonprose removed; article-opening prose excerpt selected; whitespace normalized. No translation or generated replacement text.'}


AUTOMATED = re.compile(r'lsjbot|robotskapad|botskapad|bot-generated|automatically generated article|AI-generated article|AI-genererad', re.I)


COPYRIGHT = re.compile(r'\{\{\s*(?:copyvio|copypaste|copyright violation|upphovsrättsbrott)\b', re.I)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def words(text):
    return len(text.split())


def save_retry(path, value):
    """Retry transient Windows file locks without truncating the last good file."""
    for attempt in range(6):
        try:
            save(path,value)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * 2**attempt)


def clean_prose(html):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.select_one('.mw-parser-output') or soup
    for node in body.select('table, ul, ol, dl, figure, sup, style, script, .hatnote, .shortdescription, .thumb, .navbox, .metadata, .ambox, .reference, .reflist, .mw-editsection, .noprint'):
        node.decompose()
    paragraphs = []
    for node in body.find_all(['h2','h3','p']):
        if node.name != 'p':
            if re.match(r'^(references|notes|sources|bibliography|external links|further reading|see also|källor|referenser|noter|litteratur|externa länkar|se även)\b',node.get_text(' ',strip=True),re.I):
                break
            continue
        text = unicodedata.normalize('NFC', re.sub(r'\s+', ' ', node.get_text(' ', strip=True))).strip()
        # Inline HTML boundaries do not need an extra space before punctuation.
        text = re.sub(r'\s+([,.;:!?\)])', r'\1', text)
        nonspace = [c for c in text if not c.isspace()]
        if (words(text) >= 20 and nonspace and sum(c.isalpha() for c in nonspace)/len(nonspace) >= .6
                and re.search(r'[.!?]', text)):
            paragraphs.append(text)
    full = '\n\n'.join(paragraphs)
    if words(full) < RULES['min_words']:
        raise ValueError('insufficient_prose')
    boundaries = [m.end() for m in re.finditer(r'[.!?](?:[”"\u2019\)])?(?=\s|$)', full)]
    choices = [(end, words(full[:end])) for end in boundaries if RULES['min_words'] <= words(full[:end]) <= RULES['max_words']]
    if not choices:
        raise ValueError('no_sentence_boundary_in_word_budget')
    end, count = next(((end,count) for end,count in choices if count >= RULES['target_words']), choices[-1])
    return full[:end].strip(), {'cleaned_article_prose_words': words(full), 'words': count, 'excerpt': end < len(full)}


def attribution_notices(html):
    """Preserve visible imported-text credit notices and their source links separately."""
    soup = BeautifulSoup(html, 'html.parser')
    matches = []
    pattern = re.compile(r'incorporates.*text|text.*incorporated|public domain|offentlig egendom|text.*hämtad|text.*licens|creative commons|cc.by.sa', re.I)
    for node in soup.find_all(['p','li','div']):
        text = re.sub(r'\s+', ' ', node.get_text(' ', strip=True))
        if len(text) <= 3000 and pattern.search(text):
            # Prefer innermost matching notice to avoid duplicate containing divs.
            if any(pattern.search(re.sub(r'\s+', ' ', child.get_text(' ', strip=True))) for child in node.find_all(['p','li','div'])):
                continue
            matches.append({'text':text,'links':[a['href'] for a in node.find_all('a',href=True)]})
    return matches


class API:
    def __init__(self, cache, user_agent, delay=1.0):
        self.cache, self.delay, self.last = cache, max(1.0,delay), 0.
        self.session = requests.Session()
        self.session.headers.update({'User-Agent':user_agent})
        self.events = []

    def get(self, language, params, cache_tag=''):
        params = dict(format='json',formatversion=2,maxlag=5,**params)
        key = digest(json.dumps([language,params,cache_tag],sort_keys=True))
        path = self.cache/(key+'.json')
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        for attempt in range(6):
            time.sleep(max(0.,self.delay-(time.monotonic()-self.last)))
            self.last = time.monotonic()
            response = None
            try:
                response = self.session.get(f'https://{language}.wikipedia.org/w/api.php',params=params,timeout=(15,60))
                response.raise_for_status()
                payload = response.json()
                if 'error' in payload:
                    raise RuntimeError(f"API error: {payload['error']}")
                value = {'retrieved_at':timestamp(),'url':response.url,'data':payload}
                save_retry(path,value)
                return value
            except (requests.RequestException,ValueError,RuntimeError) as exc:
                self.events.append({'at':timestamp(),'language':language,'params':params,'attempt':attempt+1,'error':str(exc)})
                wait = min(60.,2.**(attempt+1))
                if response is not None and response.headers.get('Retry-After'):
                    header = response.headers['Retry-After']
                    try:
                        retry = float(header)
                    except ValueError:
                        retry = max(0.,(parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds())
                    wait = max(wait,retry)
                if attempt == 5:
                    raise RuntimeError(f'Wikimedia request failed after 6 attempts: {exc}') from exc
                # Long Retry-After is honored; this is a background collection process.
                time.sleep(wait)


def verify_frozen(directory):
    manifest = json.loads((directory/'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Dataset manifest is incomplete')
    if 'rules' in manifest and manifest['rules'] != RULES:
        raise ValueError('Frozen dataset cleaning rules differ from current rules')
    all_hashes, all_items = set(), set()
    datasets = {}
    for lang in ['sv','en']:
        entry = manifest['datasets'][lang]
        if 'audit_file' in entry and digest((directory/entry['audit_file']).read_bytes()) != entry['audit_sha256']:
            raise ValueError(f'{lang}: collection audit hash mismatch')
        path = directory/entry['file']
        if digest(path.read_bytes()) != entry['sha256']:
            raise ValueError(f'{lang}: dataset file hash mismatch')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows) != manifest['requested_per_language'] or len(rows) != entry['documents']:
            raise ValueError(f'{lang}: incomplete document count')
        if [r['id'] for r in rows] != entry['ordered_ids'] or len({r['article_id'] for r in rows}) != len(rows):
            raise ValueError(f'{lang}: IDs changed or duplicated')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'{lang}: duplicate document IDs')
        for row in rows:
            if row['language'] != lang or row['split'] != 'test' or digest(row['text']) != row['content_sha256']:
                raise ValueError('Invalid document language, split, or hash')
            if not RULES['min_words'] <= words(row['text']) <= RULES['max_words']:
                raise ValueError('Word budget violated')
            if row['content_sha256'] in all_hashes or (row.get('wikidata_id') and row['wikidata_id'] in all_items):
                raise ValueError('Duplicate content or cross-language article topic')
            all_hashes.add(row['content_sha256'])
            if row.get('wikidata_id'):
                all_items.add(row['wikidata_id'])
            for key in ['source_url','revision_url','history_url','revision_id','retrieved_at','license','attribution','modifications']:
                if not row.get(key):
                    raise ValueError(f'Missing provenance: {key}')
            if row['license'] != LICENSE:
                raise ValueError('Missing or changed dataset license terms')
        datasets[lang] = rows
    return manifest, datasets


def collect(directory, count, user_agent, max_candidates, exclude_directory=None):
    excluded = []
    excluded_hash = None
    if exclude_directory is not None:
        directories = exclude_directory if isinstance(exclude_directory,(list,tuple)) else [exclude_directory]
        hashes = []
        for prior in directories:
            _,old = verify_frozen(prior)
            excluded.extend(doc for docs in old.values() for doc in docs)
            hashes.append(digest((prior/'dataset_manifest.json').read_bytes()))
        excluded_hash = hashes if isinstance(exclude_directory,(list,tuple)) else hashes[0]
    manifest_path = directory/'dataset_manifest.json'
    if manifest_path.exists():
        manifest,datasets = verify_frozen(directory)
        if manifest['requested_per_language'] != count:
            raise ValueError('Frozen count differs; use a different --output directory')
        if manifest.get('excluded_manifest_sha256') != excluded_hash:
            raise ValueError('Frozen exclusion corpus differs')
        print('Frozen dataset verified; no network calls or changes.',flush=True)
        return
    directory.mkdir(parents=True,exist_ok=True)
    state_path = directory/'collection_progress.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'started_at':timestamp(),'count':count,'rules':RULES,'user_agent':user_agent,'datasets':{},'api_errors':[],
        'excluded_manifest_sha256':excluded_hash}
    if state['count'] != count or state['rules'] != RULES:
        raise ValueError('Existing collection settings differ; use a different output directory')
    if state.get('excluded_manifest_sha256') != excluded_hash:
        raise ValueError('Existing exclusion settings differ')
    api = API(directory/'.collection_cache', user_agent)
    try:
        for lang in ['sv','en']:
            part = state['datasets'].setdefault(lang,{'documents':[],'audit':[],'queue':[],'batches':0})
            while len(part['documents']) < count:
                if len(part['audit']) >= max_candidates:
                    raise RuntimeError(f'{lang}: insufficient qualifying articles after {max_candidates} candidates')
                if not part['queue']:
                    part['batches'] += 1
                    response = api.get(lang,dict(action='query',generator='random',grnnamespace=0,
                        grnfilterredir='nonredirects',grnlimit=20,grnminsize=RULES['candidate_min_bytes'],
                        prop='info|pageprops|revisions',rvprop='ids|timestamp|content',rvslots='main'),
                        cache_tag=f"batch-{part['batches']}")
                    part['queue'] = sorted(response['data']['query']['pages'],key=lambda p:p.get('index',p['pageid']))
                    save_retry(state_path,state)
                page = part['queue'][0]
                audit = {'article_id':page.get('pageid'),'title':page.get('title')}
                try:
                    known = excluded+[d for item in state['datasets'].values() for d in item['documents']]
                    if any(d['language'] == lang and d['article_id'] == page['pageid'] for d in known):
                        raise ValueError('duplicate_page')
                    if page.get('ns') != 0 or 'redirect' in page or 'missing' in page:
                        raise ValueError('not_regular_article')
                    props = page.get('pageprops',{})
                    if 'disambiguation' in props or re.match(r'^(?:List of|Lists of|Lista över|Listor över)\b',page['title']):
                        raise ValueError('disambiguation_or_list')
                    qid = props.get('wikibase_item')
                    if qid and any(d.get('wikidata_id') == qid for d in known):
                        raise ValueError('duplicate_wikidata_topic')
                    revision = page['revisions'][0]
                    raw = revision['slots']['main']['content']
                    if raw.lstrip().lower().startswith(('#redirect','#omdirigering')):
                        raise ValueError('redirect_revision')
                    if AUTOMATED.search(raw):
                        raise ValueError('marked_automated_article')
                    if COPYRIGHT.search(raw):
                        raise ValueError('copyright_problem_marker')
                    parsed = api.get(lang,dict(action='parse',oldid=revision['revid'],prop='text|properties',disableeditsection=1))
                    article = parsed['data']['parse']
                    if article['revid'] != revision['revid'] or article['pageid'] != page['pageid']:
                        raise RuntimeError('Revision or page changed during parsing')
                    if 'disambiguation' in article.get('properties',{}):
                        raise ValueError('disambiguation_revision')
                    text,stats = clean_prose(article['text'])
                    h = digest(text)
                    if any(d['content_sha256'] == h for d in known):
                        raise ValueError('duplicate_content')
                    source = f"https://{lang}.wikipedia.org/wiki/{quote(page['title'].replace(' ','_'))}"
                    doc = dict(id=f"{lang}-wiki-{page['pageid']}-{revision['revid']}",language=lang,split='test',
                        title=page['title'],text=text,article_id=page['pageid'],revision_id=revision['revid'],
                        revision_timestamp=revision['timestamp'],retrieved_at=parsed['retrieved_at'],source_url=source,
                        revision_url=f"https://{lang}.wikipedia.org/w/index.php?oldid={revision['revid']}",
                        history_url=f"https://{lang}.wikipedia.org/w/index.php?curid={page['pageid']}&action=history",
                        wikidata_id=qid,content_sha256=h,license=LICENSE,attribution=f"Wikipedia contributors, {page['title']}",
                        attribution_notices=attribution_notices(article['text']),modifications=RULES['modifications'],**stats)
                    part['documents'].append(doc)
                    audit['decision'] = 'accepted'
                    print(f"{lang}: {len(part['documents'])}/{count} ({len(part['audit'])+1} candidates)",flush=True)
                except ValueError as exc:
                    audit['decision'] = str(exc)
                part['audit'].append(audit)
                part['queue'].pop(0)
                save_retry(state_path,state)
        manifest = dict(schema_version=1,status='complete',requested_per_language=count,
            started_at=state['started_at'],frozen_at=timestamp(),rules=RULES,user_agent=user_agent,
            script_sha256=digest(Path(__file__).read_bytes()),license=LICENSE,
            selection_independent_of_models=True,api_errors=state['api_errors']+api.events,datasets={},
            excluded_manifest_sha256=excluded_hash)
        for lang,part in state['datasets'].items():
            path = directory/f'eval_{lang}.jsonl'
            path.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in part['documents']),encoding='utf-8')
            audit_path = directory/f'collection_audit_{lang}.json'
            save_retry(audit_path,part['audit'])
            manifest['datasets'][lang] = dict(file=path.name,documents=len(part['documents']),sha256=digest(path.read_bytes()),
                ordered_ids=[d['id'] for d in part['documents']],words=sum(d['words'] for d in part['documents']),
                audit_file=audit_path.name,audit_sha256=digest(audit_path.read_bytes()),
                decisions=dict(Counter(a['decision'] for a in part['audit'])))
        save_retry(manifest_path,manifest)
        save_retry(directory/'collection_status.json',{'status':'complete','finished_at':timestamp()})
        verify_frozen(directory)
        print('Both datasets frozen and verified.',flush=True)
    except Exception as exc:
        state['api_errors'].extend(api.events)
        save_retry(state_path,state)
        save_retry(directory/'collection_status.json',{'status':'incomplete','error':repr(exc),'at':timestamp(),
             'counts':{lang:len(part['documents']) for lang,part in state['datasets'].items()}})
        raise


def main():
    if (ROOT/'data/datasets.zip').exists():
        raise SystemExit('Packed repository: use python run_project.py to verify frozen data, or --unpack-only to inspect it.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data')
    parser.add_argument('--count',type=int,default=150)
    parser.add_argument('--max-candidates',type=int,default=5000)
    parser.add_argument('--user-agent',default=USER_AGENT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--exclude-directory',type=Path,help='Exclude pages, Wikidata topics and text hashes from an existing frozen corpus')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.verify_only:
        verify_frozen(args.output)
        print('Frozen dataset integrity verified.')
    else:
        collect(args.output,args.count,args.user_agent,args.max_candidates,args.exclude_directory)


if __name__ == '__main__':
    main()
