# Results

The three CSVs and `figures/` expose the main 150-document-per-language evaluation.
`follow_up_summary.csv` separates the two follow-up confirmation datasets; their
plots also appear in `figures/`. Refresh with `python run_project.py --results-only`.
The code exports numerical artifacts and never edits the two author-maintained notes
in `reports/`. No PDF reports are generated.
`experiment_results.csv` includes condition/seed/language/split observations;
`method_summary.csv` contains token-weighted NLL, retention, regression and intervals;
`paired_comparisons.csv` uses candidate-minus-reference NLL (negative favors candidate).
The raw CSV `gain_ci` fields instead describe base-minus-patch gains.

`evidence.zip` contains all original detailed run directories and archived legacy
reports, with an SHA-256 manifest. Run `python run_project.py --unpack-only` to inspect
them under `.cache/work/results/` and `.cache/work/reports/legacy/`. Nothing in the
archive is relabeled as a fresh model run.

Main measurements are `wikipedia/full/`; the original synthetic study is `full/`.
Balanced diagnostics/confirmation are under `follow_up/`; dense development and
confirmation are under `dense_follow_up/`. `evaluator_audit/` includes statistical
checks, mask verification and ancestry evidence. All document losses, token counts,
seeds, model revisions, source lineage and negative outcomes are retained.

`python run_project.py --audit-project` rechecks all studies. Add `--check-weights`
to reconstruct masks from the cached pinned checkpoints without likelihood inference.
Bootstrap intervals condition on fixed checkpoints/masks and are pointwise. Corpus
disjointness does not establish training independence. See the reports for limitations.
