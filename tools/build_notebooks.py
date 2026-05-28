"""Generate the analysis notebooks from a single source.

Run from the repo root:
    python tools/build_notebooks.py

Produces:
    notebooks/01_disparities.ipynb
    notebooks/02_by_slice.ipynb
    notebooks/03_by_prompt_family.ipynb
    notebooks/04_paper_figures.ipynb
"""
import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines]}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ----------------------------- Notebook 1 -----------------------------
nb1 = [
    md(
        "# 01 — Disparities across demographic slices",
        "",
        "Reproduces Table 2 of the project page (max-minus-min disparity per metric across",
        "the four demographic slices) and saves a horizontal bar chart to `figures/disparities.pdf`.",
        "",
        "Source CSV: `analysis/big_study_scored/summary_disparities.csv`.",
    ),
    code(
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from pathlib import Path",
        "",
        "ROOT = Path('..').resolve()",
        "FIG = ROOT / 'figures'",
        "FIG.mkdir(exist_ok=True)",
    ),
    code(
        "df = pd.read_csv(ROOT / 'analysis/big_study_scored/summary_disparities.csv')",
        "df = df.sort_values('max_minus_min', ascending=True).reset_index(drop=True)",
        "df",
    ),
    md(
        "## LaTeX table (matches `experiments_appendix.tex` Table 1)",
    ),
    code(
        "label_map = {",
        "    'refusal_rate': 'Refusal rate',",
        "    'deflection_rate': 'Deflection rate',",
        "    'demeaning_kw_rate': 'Demeaning-keyword rate',",
        "    'identity_salience_rate': 'Identity-salience rate',",
        "    'religion_mention_rate': 'Religion-mention rate',",
        "    'pilot_harm_rate': 'Pilot harm rate',",
        "    'stereotype_kw_rate': 'Stereotype-keyword rate',",
        "    'title_mention_rate': 'Title-mention rate',",
        "}",
        "rows = [",
        "    f'    {label_map[r.metric]:<24} & {r.max_minus_min:.5f} \\\\\\\\'",
        "    for r in df.itertuples()",
        "]",
        "print('\\n'.join(rows))",
    ),
    md(
        "## Horizontal bar chart",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 3.5))",
        "ax.barh(",
        "    [label_map[m] for m in df.metric],",
        "    df.max_minus_min,",
        "    color='#4c72b0',",
        ")",
        "ax.set_xlabel('Max disparity (max − min across slices)')",
        "ax.set_xlim(0, max(df.max_minus_min) * 1.15)",
        "ax.spines[['top', 'right']].set_visible(False)",
        "fig.tight_layout()",
        "fig.savefig(FIG / 'disparities.pdf')",
        "fig.savefig(FIG / 'disparities.png', dpi=150)",
        "plt.show()",
    ),
]

# ----------------------------- Notebook 2 -----------------------------
nb2 = [
    md(
        "# 02 — Slice-level rates",
        "",
        "Reproduces the per-slice tables from the appendix (Tables 2, 5, 6 / project page Tables 3, 6, 7):",
        "",
        "- Per-slice rates, averaged over occupation / family / paraphrase / decoding / seed.",
        "- Per-slice × occupation breakdown.",
        "- Per-slice × decoding breakdown.",
    ),
    code(
        "import pandas as pd",
        "from pathlib import Path",
        "",
        "ROOT = Path('..').resolve()",
        "DATA = ROOT / 'analysis/big_study_scored'",
    ),
    md("## Per-slice"),
    code(
        "by_slice = pd.read_csv(DATA / 'summary_by_slice.csv')",
        "by_slice",
    ),
    md("## Per-slice × occupation"),
    code(
        "by_occ = pd.read_csv(DATA / 'summary_by_slice_occupation.csv')",
        "by_occ",
    ),
    md("## Per-slice × decoding"),
    code(
        "by_dec = pd.read_csv(DATA / 'summary_by_slice_decoding.csv')",
        "by_dec",
    ),
    md(
        "## Quick check — does any rate disparity exceed 5% within an occupation?",
        "",
        "The occupation breakdown surfaces the largest local disparities (e.g. nurse role,",
        "stereotype-keyword rate across slices).",
    ),
    code(
        "metric = 'stereotype_kw_rate'",
        "by_occ.pivot(index='occupation', columns='full_slice_id', values=metric).round(3)",
    ),
]

# ----------------------------- Notebook 3 -----------------------------
nb3 = [
    md(
        "# 03 — Prompt-family sensitivity (headline result)",
        "",
        "This is the central figure of the paper: the worst-slice stereotype-keyword rate",
        "across the four prompt families. The same model crosses a hypothetical 5% audit",
        "threshold on every slice under F2 (story continuation) and on no slice under F4",
        "(HR memo).",
        "",
        "Reproduces appendix Table 3 (per-slice × family) and Table 4 (joint factorial",
        "with decoding) plus the page's worst-slice plot.",
    ),
    code(
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "from pathlib import Path",
        "",
        "ROOT = Path('..').resolve()",
        "DATA = ROOT / 'analysis/big_study_scored'",
        "FIG = ROOT / 'figures'",
        "FIG.mkdir(exist_ok=True)",
    ),
    md("## Per-slice × prompt family"),
    code(
        "by_fam = pd.read_csv(DATA / 'summary_by_slice_prompt.csv')",
        "by_fam",
    ),
    md("## Worst-slice stereotype rate per family"),
    code(
        "ws = by_fam.groupby('prompt_family')['stereotype_kw_rate'].max().reset_index()",
        "ws['min_slice'] = by_fam.groupby('prompt_family')['stereotype_kw_rate'].min().values",
        "ws",
    ),
    md("## Headline figure"),
    code(
        "families = ['F1', 'F2', 'F3', 'F4']",
        "slices = ['man_Christian', 'man_Muslim', 'woman_Christian', 'woman_Muslim']",
        "slice_labels = ['M × Christian', 'M × Muslim', 'F × Christian', 'F × Muslim']",
        "",
        "pivot = by_fam.pivot(index='prompt_family', columns='full_slice_id', values='stereotype_kw_rate')",
        "pivot = pivot.loc[families, slices]",
        "",
        "fig, ax = plt.subplots(figsize=(6.5, 3.5))",
        "x = np.arange(len(families))",
        "w = 0.2",
        "colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']",
        "for i, (sl, lbl) in enumerate(zip(slices, slice_labels)):",
        "    ax.bar(x + (i - 1.5) * w, pivot[sl], width=w, label=lbl, color=colors[i])",
        "ax.axhline(0.05, color='black', linestyle=':', linewidth=1, label='5% threshold')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels(families)",
        "ax.set_xlabel('Prompt family')",
        "ax.set_ylabel('Stereotype-keyword rate')",
        "ax.legend(loc='upper right', frameon=False, fontsize=8)",
        "ax.spines[['top', 'right']].set_visible(False)",
        "fig.tight_layout()",
        "fig.savefig(FIG / 'protocol_flip.pdf')",
        "fig.savefig(FIG / 'protocol_flip.png', dpi=150)",
        "plt.show()",
    ),
    md(
        "## Joint factorial: slice × family × decoding",
        "",
        "The variance across cells is larger than the variance across slices within a cell.",
    ),
    code(
        "joint = pd.read_csv(DATA / 'summary_by_slice_prompt_decoding.csv')",
        "joint",
    ),
    md(
        "## Seed sensitivity",
        "",
        "Within the finest factorial cell (`summary_by_protocol.csv`, n=5 seeds per row),",
        "the worst-slice stereotype rate over fixed-prompt resamples lies in [0.094, 0.125],",
        "matching the figure quoted in §5 of the paper.",
    ),
    code(
        "protocol = pd.read_csv(DATA / 'summary_by_protocol.csv')",
        "# Hold prompt fixed (one paraphrase per family), aggregate across slices to find worst-slice",
        "fixed = protocol[protocol['paraphrase_id'] == 'p1']",
        "ws_by_cell = fixed.groupby(['prompt_family', 'occupation', 'decoding_regime'])['stereotype_kw_rate'].max()",
        "print(f'Worst-slice stereotype rate over fixed-prompt cells: [{ws_by_cell.min():.3f}, {ws_by_cell.max():.3f}]')",
    ),
]

# ----------------------------- Notebook 4 -----------------------------
nb4 = [
    md(
        "# 04 — Project-page teaser figure",
        "",
        "Builds the hero figure for the project webpage: a side-by-side panel showing",
        "(left) max disparity across slices on each metric, (right) the worst-slice",
        "stereotype rate by prompt family with the 5% audit threshold.",
        "",
        "Output: `figures/teaser.pdf` and `figures/teaser.png`.",
    ),
    code(
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "from pathlib import Path",
        "",
        "ROOT = Path('..').resolve()",
        "DATA = ROOT / 'analysis/big_study_scored'",
        "FIG = ROOT / 'figures'",
        "FIG.mkdir(exist_ok=True)",
        "",
        "plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})",
    ),
    code(
        "disp = pd.read_csv(DATA / 'summary_disparities.csv').sort_values('max_minus_min')",
        "fam = pd.read_csv(DATA / 'summary_by_slice_prompt.csv')",
        "",
        "label_map = {",
        "    'refusal_rate': 'Refusal',",
        "    'deflection_rate': 'Deflection',",
        "    'demeaning_kw_rate': 'Demeaning kw.',",
        "    'identity_salience_rate': 'Identity salience',",
        "    'religion_mention_rate': 'Religion mention',",
        "    'pilot_harm_rate': 'Pilot harm',",
        "    'stereotype_kw_rate': 'Stereotype kw.',",
        "    'title_mention_rate': 'Title mention',",
        "}",
        "families = ['F1', 'F2', 'F3', 'F4']",
        "slices = ['man_Christian', 'man_Muslim', 'woman_Christian', 'woman_Muslim']",
        "slice_labels = ['M × Christian', 'M × Muslim', 'F × Christian', 'F × Muslim']",
        "colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']",
        "",
        "fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [1, 1.1]})",
        "",
        "# Left: max disparities",
        "axL.barh([label_map[m] for m in disp.metric], disp.max_minus_min, color='#4c72b0')",
        "axL.set_xlabel('Max disparity across demographic slices')",
        "axL.set_title('Slice disparities look small', loc='left', fontweight='bold')",
        "",
        "# Right: worst-slice stereotype rate per family",
        "pivot = fam.pivot(index='prompt_family', columns='full_slice_id', values='stereotype_kw_rate')",
        "pivot = pivot.loc[families, slices]",
        "x = np.arange(len(families))",
        "w = 0.2",
        "for i, (sl, lbl) in enumerate(zip(slices, slice_labels)):",
        "    axR.bar(x + (i - 1.5) * w, pivot[sl], width=w, label=lbl, color=colors[i])",
        "axR.axhline(0.05, color='black', linestyle=':', linewidth=1, label='5% audit threshold')",
        "axR.set_xticks(x)",
        "axR.set_xticklabels(families)",
        "axR.set_xlabel('Prompt family')",
        "axR.set_ylabel('Stereotype-keyword rate')",
        "axR.set_title('… until you change the prompt family', loc='left', fontweight='bold')",
        "axR.legend(loc='upper right', frameon=False, fontsize=7.5)",
        "",
        "fig.tight_layout()",
        "fig.savefig(FIG / 'teaser.pdf')",
        "fig.savefig(FIG / 'teaser.png', dpi=200)",
        "plt.show()",
    ),
]


if __name__ == "__main__":
    nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
    write_nb(nb_dir / "01_disparities.ipynb", nb1)
    write_nb(nb_dir / "02_by_slice.ipynb", nb2)
    write_nb(nb_dir / "03_by_prompt_family.ipynb", nb3)
    write_nb(nb_dir / "04_paper_figures.ipynb", nb4)
