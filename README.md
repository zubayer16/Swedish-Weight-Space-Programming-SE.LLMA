# SE.LLMA Weight-Space Programming

This project tests whether selecting a small portion of weight updates can preserve
Swedish performance while limiting English regression.

## Run

Use Python 3.12:

```sh
python -m pip install -r requirements.txt
python run_project.py
```

The script unpacks the saved data, checks completed results, and exports tables and plots.
Completed experiments are reused.

- Check tests and results: `python run_project.py --audit-project`
- Refresh tables and plots: `python run_project.py --results-only`
- Inspect archived data: `python run_project.py --unpack-only`

## Experiments completed

1. **Sparse selection:** After an eight-text pilot per language, we compared
   layer-aware selection with random, DARE, magnitude and layer-random controls at
   10% and 30% retention on 150 Wikipedia excerpts per language. Layer-aware selection
   lost to random on Swedish prediction at both budgets.
2. **Balanced allocation:** We tested keeping the largest updates at the same rate
   within every layer/group. On 50 new excerpts per language, balanced 30% improved
   Swedish over the original rule, but retained only 7.9% of the full gain and still
   worsened English prediction.
3. **Dense adaptation:** We compared eight update strengths on development texts,
   then evaluated the selected strength on another 50 new excerpts per language.
   Selection chose full adaptation (α = 1), not an intermediate-strength solution.

All experiments measured token-weighted NLL: lower is better. The original sparse
selection hypothesis was not supported; the follow-ups did not resolve the
Swedish–English trade-off.

## Reports and files

- [Final research note - PDF](reports/research_note.pdf) and [two-page Markdown summary](reports/research_note.md)
- These notes are maintained by the author. The code exports results and plots, checks the notes, and never rewrites them.
- `src/`, `tests/`, `configs/`: code, tests, and settings.
- `data/`, `results/`: frozen datasets, tables, plots, and archived evidence.

Archives unpack into `.cache/work/`. Wikipedia attribution and licensing are in
[data/LICENSE.md](data/LICENSE.md).

The exact checkpoint ancestry remains unverified,
and evaluation texts may overlap with model training data.

Earlier reports, plans and audits are preserved inside `results/evidence.zip`.

### AI Tool Usage

AI tools were used to assist with debugging, test code generation, example codes and preliminary review of experimental results. Generated suggestions were evaluated for technical correctness and consistency with the recorded findings.
