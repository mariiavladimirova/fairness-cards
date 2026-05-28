# Project page: Fairness Cards

Source for the ICML 2026 project page for *Fairness Failure in Generative Models is an Evaluation Problem* by Mariia Vladimirova, Jean-Yves Franceschi, and Thibaut Issenhuth (Criteo AI Lab). Built on top of the [Nerfies](https://github.com/nerfies/nerfies.github.io) template.

## Page sections

1. Hero — title, authors, ICML 2026 badge, code + BibTeX buttons.
2. Abstract.
3. Teaser figure — slice disparities vs. prompt-family sensitivity.
4. Methodology snapshot — what a Fairness Card discloses.
5. Table 1 — comparison with prior documentation frameworks (Model Cards, Datasheets, AI FactSheets, Reproducibility Standards).
6. Discussion — what the Qwen2.5-7B-Instruct audit shows (per-cell tables live in the paper appendix and the code repo, not on this page).
7. Filled Fairness Card — the concrete disclosure for the Qwen audit.
8. Operationalisation roadmap — what tooling would turn Fairness Cards into a cumulative practice.
9. BibTeX.
10. Footer.

## What's here

```
fairness-cards-page/
├── index.html                     The page
├── .nojekyll                      Tells GitHub Pages to serve files verbatim
└── static/
    ├── css/                       Bulma, FontAwesome, Nerfies custom CSS
    ├── js/                        FontAwesome icons
    └── images/                    Teaser and result figures (from notebooks)
```

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Publishing on GitHub Pages

Push this directory to a repo and enable Pages on the `main` branch, root folder. No build step required.

## Updating the figures

The teaser and result figures are produced by the analysis notebooks in the [code release](https://github.com/mariiavladimirova/fairness-cards). To refresh:

```bash
# in the code repo
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
# then copy the PNGs back here
cp ../fairness-cards/figures/{teaser,protocol_flip,disparities}.png static/images/
```

## Pending before publication

- Paper PDF and arXiv buttons are currently absent — add them to the hero block once a stable URL exists (e.g. PMLR or arXiv link).
