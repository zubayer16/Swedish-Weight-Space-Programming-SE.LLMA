"""Export result tables and figures; research notes are edited by the author."""
import argparse
import json
import re
import shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reported_values(text):
    """Extract three-decimal measurements, including sentence-final numbers."""
    return set(re.findall(r'(?<![\w.])\d+\.\d{3}(?!\d)',text))


def export_results(root):
    """Refresh numerical artifacts without writing to reports/ or README."""
    main = root/'results/wikipedia/full'
    output = root/'results'
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    for name in ['experiment_results.csv','method_summary.csv','paired_comparisons.csv']:
        shutil.copyfile(main/name,output/name)
    summary = pd.read_csv(main/'method_summary.csv')
    if not (main/'figures/wikipedia_nll.png').exists():
        (main/'figures').mkdir(exist_ok=True)
        plot_expanded(summary,main/'figures')
    for name in ['weight_distribution.png','wikipedia_nll.png','wikipedia_paired_intervals.png']:
        if (main/'figures'/name).exists():
            shutil.copyfile(main/'figures'/name,figures/name)
    follow_ups = []
    for name in ['follow_up','dense_follow_up']:
        directory = output/name
        table = directory/'confirmation/method_summary.csv'
        if not table.exists():
            continue
        frame = pd.read_csv(table)
        frame.insert(0,'evaluation_set',name)
        if name == 'dense_follow_up':
            selected = json.loads((directory/'selection.json').read_text())['selection']['alpha']
            frame['selected_dense_alpha'] = selected
        follow_ups.append(frame)
        # Historical figures remain byte-identical inside their original run.
        # New views are written only to the convenient results/figures folder.
        fig,ax = plt.subplots(figsize=(7,4))
        for (method,fraction),rows in frame.groupby(['method','fraction']):
            by_language = rows.set_index('language')
            ax.scatter(by_language.loc['en','nll'],by_language.loc['sv','nll'])
            ax.annotate(f'{method} {fraction:.0%}',
                        (by_language.loc['en','nll'],by_language.loc['sv','nll']),fontsize=7,
                        xytext=(4,3),textcoords='offset points')
        ax.set(xlabel='English NLL',ylabel='Swedish NLL',title=f'{name}: confirmation results')
        fig.tight_layout()
        fig.savefig(figures/f'{name}.png',dpi=160)
        plt.close(fig)
    if follow_ups:
        pd.concat(follow_ups,ignore_index=True).to_csv(output/'follow_up_summary.csv',index=False)
    print('Exported result tables and plots. Author-written notes were not changed.')


def validate_notes(root):
    """Check human-maintained note tables against measured results, without rewriting them."""
    from .datasets import digest
    note = root/'reports/research_note.md'
    pdf = root/'reports/research_note.pdf'
    markdown = note.read_text(encoding='utf-8')
    main = pd.read_csv(root/'results/wikipedia/full/method_summary.csv')

    def identify(label):
        label = label.lower().strip()
        if label.startswith('base'):
            return 'base',0.
        if label.startswith(('swedish checkpoint','full checkpoint')):
            return 'adapted',1.
        fraction = .1 if '10%' in label else .3
        if label.startswith('random'):
            return 'random',fraction
        if label.startswith(('global magnitude','magnitude')):
            return 'magnitude',fraction
        if label.startswith(('original layer','layer-aware')):
            return 'layer',fraction
        if label.startswith('balanced'):
            return 'balanced',fraction
        if label.startswith('dare'):
            return 'dare',fraction
        return None

    def check_row(label,sv,en,frame):
        method,fraction = identify(label)
        for lang,value in [('sv',sv),('en',en)]:
            row = frame[(frame.method == method)&(frame.fraction == fraction)&(frame.language == lang)]
            assert len(row) == 1 and float(value) == round(float(row.iloc[0].nll),3), (label,lang,value)

    md_rows = 0
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip('|').split('|')]
        if line.startswith('|') and identify(cells[0]) is not None:
            check_row(cells[0],cells[1],cells[2],main)
            md_rows += 1
    assert md_rows >= 5, 'The short note must include the main baseline and sparse comparisons'
    assert len(markdown.split()) <= 900, 'Keep the Markdown note concise (two-page target)'
    # Read the author's final PDF for consistency checks, never regenerate it.
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    pdf_text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    assert reported_values(markdown) <= reported_values(pdf_text), 'A numerical claim in the summary is absent from the final PDF'
    assert '\u2014' not in markdown and '\u2014' not in pdf_text, 'Use no em dashes in the notes'
    return {'author_maintained':True,'markdown_words':len(markdown.split()),
            'markdown_table_rows_verified':md_rows,'summary_values_present_in_final_pdf':True,
            'pdf_pages':len(reader.pages),
            'markdown_sha256':digest(note.read_bytes()),'pdf_sha256':digest(pdf.read_bytes())}


def plot_weights(frame, output):
    grouped = frame.groupby('group', sort=True).squared_norm_share.sum()
    fig, ax = plt.subplots(figsize=(10,3))
    grouped.plot.bar(ax=ax)
    ax.set_ylabel('Squared norm share')
    ax.set_xlabel('Unique parameter group')
    fig.tight_layout()
    fig.savefig(output/'weight_distribution.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(frame, output):
    f = frame[(frame.split == 'test') & (frame.language == 'sv')]
    fig, ax = plt.subplots(figsize=(8,4))
    for method, rows in f.groupby('method'):
        means = rows.groupby('retained_fraction').nll.mean()
        ax.plot(means.index, means.values, 'o-', label=method)
    ax.set(xlabel='Retained fraction of unique parameters',ylabel='Swedish NLL (lower is better)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'swedish_nll.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def plot_expanded(summary, output):
    """Uncertainty across documents, conditional on the fixed mask seeds."""
    import numpy as np
    order = [('base',0.),('adapted',1.)]+[(m,f) for f in [.1,.3]
            for m in ['random','dare','magnitude','layer','layer_random']]
    fig,axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
    for ax,language in zip(axes,['sv','en']):
        rows = summary[summary.language == language].set_index(['method','fraction']).loc[order]
        value = rows.delta_nll_vs_base.to_numpy()
        errors = np.vstack([value-rows.delta_ci_low.to_numpy(),rows.delta_ci_high.to_numpy()-value])
        ax.errorbar(value,np.arange(len(rows)),xerr=np.maximum(errors,0),fmt='o',capsize=3,color='#246d91')
        ax.axvline(0,color='grey',lw=1)
        ax.set_xscale('symlog',linthresh=.15)
        ax.set_title('Swedish' if language == 'sv' else 'English')
        ax.set_xlabel('NLL change from base (symlog); lower is better')
        ax.grid(axis='x',alpha=.25)
    axes[0].set_yticks(np.arange(len(order)),[f'{m} {f:.0%}' for m,f in order])
    axes[0].invert_yaxis()
    fig.suptitle('Frozen Wikipedia evaluation: paired document bootstrap 95% intervals')
    fig.tight_layout()
    fig.savefig(output/'wikipedia_paired_intervals.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,language in zip(axes,['sv','en']):
        for method,rows in summary[summary.language == language].groupby('method'):
            rows = rows.sort_values('fraction')
            errors = np.maximum(np.vstack([rows.nll-rows.nll_ci_low,rows.nll_ci_high-rows.nll]),0)
            ax.errorbar(rows.fraction,rows.nll,yerr=errors,fmt='o-',capsize=3,label=method)
        ax.set(title='Swedish' if language == 'sv' else 'English',xlabel='Retained parameter fraction',ylabel='Token-weighted NLL')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'wikipedia_nll.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    export_results(Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
