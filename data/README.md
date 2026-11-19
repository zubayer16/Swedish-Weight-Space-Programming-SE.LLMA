# Frozen evaluation data

`datasets.zip` contains the original files at their original paths:

- `data/eval_sv.jsonl`, `data/eval_en.jsonl`, and `data/dataset_manifest.json`: 150 articles per language.
- `data/follow_up/`: 50 new articles per language for balanced-allocation confirmation.
- `data/dense_follow_up/`: 50 additional articles per language for dense confirmation.
- Collection audits, full cleaning/sampling documentation and manifests for each corpus.
- `data/sample_sv.jsonl` and `data/sample_en.jsonl`: the unchanged 11 legacy examples per language.

Run `python run_project.py --unpack-only` from the repository root to verify and
inspect these files under `.cache/work/data/`. Normal workflows unpack automatically.
Every archive entry has an SHA-256 checksum. The frozen manifests additionally
verify exact document order, text hashes, attribution and license information.

The Wikimedia collector uses consistent 100–200-word prose excerpts, rate limits,
an identifiable User-Agent, retries and explicit incomplete-sample status. Full
rules and rejection audits are inside the archive. No replacement text is generated
or translated. Exclusively human authorship and training independence are unproven.
Follow-up corpora exclude earlier page IDs, Wikidata topics and content hashes.

See `LICENSE.md`. Preserve each document's contributor attribution, article,
revision/history URLs, CC BY-SA 4.0 license and transformation notices when sharing
the archive or extracted text. This license also applies to archived Wikipedia text.
