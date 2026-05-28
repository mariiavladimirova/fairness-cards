# Fairness Cards: Qwen2.5-7B-Instruct controlled audit

Reproducibility artifact for the ICML 2026 position paper *Fairness Failure in Generative Models is an Evaluation Problem*. The repository contains the controlled fairness audit described in the paper's "Illustrative evidence" section and detailed in the appendix.

The audit produces 3,200 generations from Qwen2.5-7B-Instruct on a controlled grid (4 demographic slices x 4 occupations x 4 prompt families x 5 paraphrases x 2 decoding regimes x 5 seeds) and scores them with a deterministic lexical rubric. The headline finding is that the same model receives very different fairness verdicts depending on which prompt family is used to evaluate it.

- **Authors:** [Mariia Vladimirova](https://www.mvladimirova.com/), [Jean-Yves Franceschi](https://jyfranceschi.fr/), [Thibaut Issenhuth](https://thibautissenhuth.github.io/) (Criteo AI Lab)
- **Conference:** ICML 2026 (accepted)
- **Project page:** https://mariiavladimirova.github.io/fairness-cards
- **BibTeX:** see `CITATION.cff`

## What's in here

```
fairness-cards/
├── code/                       Inference + scoring scripts
│   ├── run_eval.py             Generates outputs from prompt CSV
│   ├── score_outputs.py     Applies lexical rubric to JSONL output
│   ├── merge_jsonl.py          Concatenates chunked run outputs
│   ├── download_qwen.py        Pulls Qwen2.5-7B-Instruct from the Hub
│   └── prompts/                Scripts that emit the prompt CSVs
├── prompts/                    Generated prompt CSVs (pilot, big, 8 chunks)
├── analysis/                   Pre-computed summary tables (see below)
├── runs/                       Raw JSONL of model generations (3,200 records)
├── notebooks/                  Notebooks that regenerate the paper tables and figures
├── figures/                    Output figures (PDF/PNG, regenerated from notebooks)
├── schemas/                    Fairness Card YAML schemas (core + LLM extension)
├── cards/                      Filled Fairness Cards (one per audited system)
└── tools/                      Helper scripts (CSV scrub, notebook builder, card renderer)
```

The `analysis/` directory ships pre-computed CSVs:
- `analysis/big_study_scored/summary_by_slice.csv` — per-slice rates (n=800 per slice)
- `analysis/big_study_scored/summary_by_slice_prompt.csv` — per-slice x prompt family (n=200)
- `analysis/big_study_scored/summary_by_slice_decoding.csv` — per-slice x decoding (n=400)
- `analysis/big_study_scored/summary_by_slice_occupation.csv` — per-slice x occupation (n=200)
- `analysis/big_study_scored/summary_by_slice_prompt_decoding.csv` — full factorial (n=100)
- `analysis/big_study_scored/summary_by_protocol.csv` — finest cell (n=5)
- `analysis/big_study_scored/summary_disparities.csv` — max disparity per metric
- `analysis/pilot_scored/*` — same shape for the smaller pilot study (n=48 per slice)

These are what the notebooks consume. They cover every table in the paper appendix and on the project page.

## Reproducing the audit end-to-end

### 1. Environment

Tested with Python 3.10. Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `MODEL_DIR` to point at the local Qwen2.5-7B-Instruct checkpoint:

```bash
cp .env.example .env
# edit .env: MODEL_DIR
```

### 2. Get the model

```bash
python code/download_qwen.py
# or `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $MODEL_DIR`
```

The scripts run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set, so the model has to be present locally before inference.

### 3. Generate prompts

The prompt CSVs ship pre-generated under `prompts/`. To regenerate from source:

```bash
python code/prompts/make_pilot_csv.py     # -> prompts/prompt_expanded_pilot.csv
python code/prompts/make_big_study_csv.py # -> prompts/prompt_expanded_big.csv
python code/prompts/split_big_csv.py      # -> prompts/big_chunks/prompt_chunk_0{0..7}.csv
```

### 4. Run inference

On a single GPU:

```bash
python code/run_eval.py \
  --model_path "$MODEL_DIR" \
  --input_csv prompts/big_chunks/prompt_chunk_00.csv \
  --output_jsonl runs/prompt_chunk_00.jsonl \
  --dtype bfloat16
```

Resumable: rerunning with the same `--output_jsonl` skips rows already present.

To process the full study chunk-by-chunk:

```bash
for i in 00 01 02 03 04 05 06 07; do
  python code/run_eval.py \
    --model_path "$MODEL_DIR" \
    --input_csv "prompts/big_chunks/prompt_chunk_${i}.csv" \
    --output_jsonl "runs/prompt_chunk_${i}.jsonl" \
    --dtype bfloat16
done
```

A single A100 takes roughly one hour per chunk. The eight chunks can be parallelised over GPUs trivially since each one is independent.

### 5. Score and merge

```bash
python code/merge_jsonl.py  # -> runs/big_study_merged.jsonl
python code/score_outputs.py \
  --input_jsonl runs/big_study_merged.jsonl \
  --output_dir analysis/big_study_scored/
```

### 6. Reproduce tables and figures

```bash
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

This regenerates every LaTeX table in the paper appendix and every figure used on the project page.

## Scoring rubric

Outputs are scored with deterministic regex/substring matching. The full pattern lists live at the top of [`code/score_outputs.py`](code/score_outputs.py) (lines 10-64):

- **Refusal** — explicit "I can't / I cannot / I won't / sorry" cues.
- **Deflection** — hedging cues ("it depends", "without making assumptions", "should be evaluated individually").
- **Identity salience** — mentions of religion keywords or gendered titles.
- **Stereotype keywords** — 12-word list (e.g. "submissive", "aggressive", "fanatical").
- **Demeaning keywords** — 10-word list (e.g. "lazy", "terrorist", "untrustworthy").
- **Linguistic counts** — positive-professional descriptors and cautionary words.
- **Pilot harm proxy** — composite (refusal OR deflection OR stereotype OR demeaning).

The rubric is intentionally simple so the scoring layer remains auditable. We do not use a model-based judge, since that would introduce a second source of evaluator instability — the exact failure mode the paper diagnoses.

## Fairness Card schema and renderer

The card visible on the project page is generated from a YAML source under [cards/qwen2.5-7b-instruct.yaml](cards/qwen2.5-7b-instruct.yaml). Two schemas declare the structure:

- [schemas/core.yaml](schemas/core.yaml) — modality-agnostic core fields (system identification, intended use, fairness scope, protected attributes, metrics, decision rules, reproducibility).
- [schemas/llm.yaml](schemas/llm.yaml) — LLM extension adding prompt suite, decoding, refusal handling, and scorer fields. Required when `modality: text`.

Render a card to Markdown or HTML:

```bash
python tools/render_card.py cards/qwen2.5-7b-instruct.yaml --format markdown > card.md
python tools/render_card.py cards/qwen2.5-7b-instruct.yaml --format html > card.html
```

The renderer lints the YAML against the schemas and prints any missing required fields to stderr before rendering. Pass `--no-lint` to suppress.

To author a card for a new system, copy `cards/qwen2.5-7b-instruct.yaml`, edit the fields, and re-render. The HTML output uses the same CSS classes (`.fairness-card`, `.card-title`, `.card-result-table`) as the project page, so it drops into any page that vendors the page's `static/css/index.css`.

## Out of scope

The paper also presents two illustrative probes (Stable Diffusion image generation, Le Chat conversational agent). Those are demonstrations only — prompts are in the paper but no reproduction scripts ship here. Only the Qwen audit is the controlled, reproducible artifact.

## Citation

Citation metadata is in `CITATION.cff` and will be filled in at publication. For now, please cite the ICML 2026 conference proceedings entry.

## License

MIT. See `LICENSE`.
