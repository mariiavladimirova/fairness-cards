"""Render a Fairness Card YAML to HTML or Markdown.

The HTML output matches the styling used on the project page; the Markdown
output is a stand-alone document suitable for embedding in a README or
appendix.

Usage:
    python tools/render_card.py cards/qwen2.5-7b-instruct.yaml --format html > card.html
    python tools/render_card.py cards/qwen2.5-7b-instruct.yaml --format markdown > card.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import yaml


REQUIRED_TOP_LEVEL = [
    "card_version", "modality", "system", "intended_use", "fairness_scope",
    "protected_attributes", "metrics", "decision_rules", "reproducibility",
]
REQUIRED_LLM = ["prompt_suite", "decoding", "refusal_handling", "scorer"]


def lint(card: dict) -> list[str]:
    """Return a list of human-readable problems with the card (empty if clean)."""
    errors: list[str] = []
    for k in REQUIRED_TOP_LEVEL:
        if k not in card:
            errors.append(f"missing required top-level field: {k!r}")
    if card.get("modality") == "text":
        for k in REQUIRED_LLM:
            if k not in card:
                errors.append(f"text-modality card missing LLM-extension field: {k!r}")
    # Spot-checks for fields with structure.
    if isinstance(card.get("system"), dict):
        for k in ("name", "version"):
            if not card["system"].get(k):
                errors.append(f"system.{k} is missing or empty")
    if isinstance(card.get("intended_use"), dict):
        for k in ("in_scope", "out_of_scope"):
            if not card["intended_use"].get(k):
                errors.append(f"intended_use.{k} is missing or empty")
    return errors


# ──────────────────────────── HTML rendering ─────────────────────────────

def _h(s: object) -> str:
    """Escape minimal HTML."""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _ul(items: list, render=lambda x: _h(x)) -> str:
    return "<ul>\n" + "\n".join(f"  <li>{render(x)}</li>" for x in items) + "\n</ul>"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    out = ['<table class="card-result-table">']
    out.append("  <thead><tr><th>Parameter</th><th>Value</th></tr></thead>")
    out.append("  <tbody>")
    for k, v in rows:
        out.append(f"    <tr><td>{_h(k)}</td><td>{_h(v)}</td></tr>")
    out.append("  </tbody></table>")
    return "\n".join(out)


def render_html(card: dict) -> str:
    sys_ = card.get("system", {})
    title = f"Fairness Card — {sys_.get('name', '(unnamed)')}"
    parts: list[str] = []
    parts.append('<div class="fairness-card">')
    parts.append(f'  <div class="card-title">{_h(title)}</div>')

    # System identification
    parts.append("  <h3>System identification</h3>")
    sys_items = [
        f"<b>Model / version:</b> <code>{_h(sys_.get('name', '?'))}</code>, {_h(sys_.get('version', '?'))}.",
        f"<b>Surface:</b> {_h(sys_.get('surface', '?'))}.",
    ]
    if sys_.get("precision"):
        sys_items.append(f"<b>Precision:</b> <code>{_h(sys_['precision'])}</code>.")
    if sys_.get("checkpoint_url"):
        sys_items.append(f"<b>Checkpoint:</b> <a href=\"{_h(sys_['checkpoint_url'])}\">{_h(sys_['checkpoint_url'])}</a>.")
    if sys_.get("notes"):
        sys_items.append(f"<b>Notes:</b> {_h(sys_['notes'])}")
    parts.append("  " + _ul(sys_items, render=lambda s: s))

    # Intended use
    iu = card.get("intended_use", {})
    parts.append("  <h3>Intended use / out-of-scope</h3>")
    parts.append("  <ul>")
    parts.append("    <li><b>In-scope:</b></li>")
    parts.append("    " + _ul(iu.get("in_scope", [])))
    parts.append("    <li><b>Out-of-scope:</b></li>")
    parts.append("    " + _ul(iu.get("out_of_scope", [])))
    parts.append("  </ul>")

    # Fairness scope
    fs = card.get("fairness_scope", {})
    parts.append("  <h3>Fairness scope &amp; harm model</h3>")
    sub: list[str] = []
    for k, label in [
        ("representational_harms", "Representational harms"),
        ("access_harms", "Access harms"),
        ("allocative_harms", "Allocative harms"),
        ("excluded", "Excluded"),
    ]:
        if fs.get(k):
            sub.append(f"<b>{label}:</b> {_h('; '.join(fs[k]))}.")
    parts.append("  " + _ul(sub, render=lambda s: s))

    # Protected attributes
    pa = card.get("protected_attributes", {})
    parts.append("  <h3>Protected attributes &amp; slices</h3>")
    slices = pa.get("slices", [])
    slice_strs = [", ".join(f"{k}={v}" for k, v in s.items()) for s in slices]
    pa_items = [
        f"<b>Slices ({len(slices)}):</b> {_h('; '.join(slice_strs))}.",
        f"<b>Operationalisation:</b> {_h(pa.get('operationalisation', '?'))}",
    ]
    parts.append("  " + _ul(pa_items, render=lambda s: s))

    # Prompt suite (LLM)
    ps = card.get("prompt_suite")
    if ps:
        parts.append("  <h3>Prompt suite</h3>")
        fams = ps.get("families", [])
        family_lines = [
            f"<b>{_h(f['id'])} &mdash; {_h(f.get('name', ''))}.</b> "
            f"<code>{_h(f.get('template', ''))}</code> "
            f"(&times; {len(f.get('paraphrases', []))} paraphrases)"
            for f in fams
        ]
        ps_items = family_lines + [
            f"<b>Occupations:</b> {_h(', '.join(ps.get('occupations', [])))}.",
            f"<b>Total prompts:</b> {_h(ps.get('total_prompts', '?'))}.",
            f"<b>Attribute insertion:</b> {_h(ps.get('attribute_insertion_rule', '?'))}",
        ]
        parts.append("  " + _ul(ps_items, render=lambda s: s))

    # Decoding
    dec = card.get("decoding")
    if dec:
        parts.append("  <h3>Decoding / sampling</h3>")
        rows = []
        regimes = dec.get("regimes", [])
        for r in regimes:
            rows.append((
                f"Regime: {r.get('name', '?')}",
                f"t={r.get('temperature', '?')}, top_p={r.get('top_p', '?')}, "
                f"max_new_tokens={r.get('max_new_tokens', '?')}",
            ))
        if dec.get("seeds_per_prompt") is not None:
            rows.append(("Seeds per prompt", str(dec["seeds_per_prompt"])))
        if dec.get("post_processing"):
            rows.append(("Post-processing", dec["post_processing"]))
        parts.append("  " + _kv_table(rows))

    # Refusal handling
    rh = card.get("refusal_handling")
    if rh:
        parts.append("  <h3>Refusal / deflection accounting</h3>")
        rh_items = [f"<b>Refusal:</b> {_h(rh.get('refusal_definition', '?'))}"]
        if rh.get("deflection_definition"):
            rh_items.append(f"<b>Deflection:</b> {_h(rh['deflection_definition'])}")
        rh_items.append(f"<b>Retention:</b> <code>{_h(rh.get('retention_policy', '?'))}</code>")
        parts.append("  " + _ul(rh_items, render=lambda s: s))

    # Scorer
    sc = card.get("scorer")
    if sc:
        parts.append("  <h3>Scorer</h3>")
        sc_items = [f"<b>Type:</b> <code>{_h(sc.get('type', '?'))}</code>"]
        if sc.get("patterns_file"):
            sc_items.append(f"<b>Patterns file:</b> <code>{_h(sc['patterns_file'])}</code>")
        if sc.get("judge_model"):
            sc_items.append(f"<b>Judge model:</b> {_h(sc['judge_model'])}")
        if sc.get("annotator_pool"):
            sc_items.append(f"<b>Annotator pool:</b> {_h(sc['annotator_pool'])}")
        if sc.get("thresholds"):
            ths = ", ".join(f"{k}={v}" for k, v in sc["thresholds"].items())
            sc_items.append(f"<b>Thresholds:</b> {_h(ths)}")
        parts.append("  " + _ul(sc_items, render=lambda s: s))

    # Metrics
    metrics = card.get("metrics", [])
    if metrics:
        parts.append("  <h3>Metrics</h3>")
        metric_lines = [
            f"<b>{_h(m['name'])}:</b> {_h(m.get('definition', ''))}"
            for m in metrics
        ]
        parts.append("  " + _ul(metric_lines, render=lambda s: s))

    # Decision rules
    rules = card.get("decision_rules", [])
    if rules:
        parts.append("  <h3>Decision rules</h3>")
        rule_lines = []
        for r in rules:
            line = f"<b>Rule:</b> {_h(r.get('rule', ''))}"
            if r.get("rationale"):
                line += f"<br><i>Why:</i> {_h(r['rationale'])}"
            rule_lines.append(line)
        parts.append("  " + _ul(rule_lines, render=lambda s: s))

    # Example result
    er = card.get("example_result")
    if er and er.get("rows"):
        parts.append("  <h3>Example result</h3>")
        cols = er.get("columns", [])
        out = ['<table class="card-result-table">']
        out.append("  <thead><tr>" + "".join(f"<th>{_h(c)}</th>" for c in cols) + "</tr></thead>")
        out.append("  <tbody>")
        for row in er["rows"]:
            out.append("    <tr>" + "".join(f"<td>{_h(c)}</td>" for c in row) + "</tr>")
        out.append("  </tbody></table>")
        parts.append("  " + "\n".join(out))

    # Reproducibility
    rep = card.get("reproducibility", {})
    if rep:
        parts.append("  <h3>Reproducibility artifacts</h3>")
        rep_items = []
        if rep.get("code_url"):
            rep_items.append(f"<b>Code:</b> <a href=\"{_h(rep['code_url'])}\">{_h(rep['code_url'])}</a>")
        if rep.get("license"):
            rep_items.append(f"<b>License:</b> {_h(rep['license'])}")
        if rep.get("data_release"):
            rep_items.append(f"<b>Data release:</b> <code>{_h(rep['data_release'])}</code>")
        if rep.get("seeds"):
            rep_items.append(f"<b>Seeds:</b> {_h(rep['seeds'])}")
        if rep.get("total_samples") is not None:
            rep_items.append(f"<b>Total samples:</b> {_h(rep['total_samples'])}")
        if rep.get("regen_command"):
            rep_items.append(f"<b>Regenerate:</b> <code>{_h(rep['regen_command'])}</code>")
        if card.get("evaluation_date"):
            rep_items.append(f"<b>Evaluation date:</b> {_h(card['evaluation_date'])}")
        parts.append("  " + _ul(rep_items, render=lambda s: s))

    parts.append("</div>")
    return "\n".join(parts)


# ──────────────────────────── Markdown rendering ────────────────────────

def _md_list(items: list[str], indent: str = "") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)


def render_markdown(card: dict) -> str:
    sys_ = card.get("system", {})
    out: list[str] = []
    out.append(f"# Fairness Card — {sys_.get('name', '(unnamed)')}")
    out.append("")
    out.append(f"_Card version `{card.get('card_version', '?')}`, modality `{card.get('modality', '?')}`._")
    if card.get("evaluation_date"):
        out.append(f"_Evaluation date: {card['evaluation_date']}._")
    out.append("")

    # System
    out.append("## System identification")
    sys_lines = [
        f"**Model / version:** `{sys_.get('name', '?')}`, {sys_.get('version', '?')}",
        f"**Surface:** `{sys_.get('surface', '?')}`",
    ]
    if sys_.get("precision"):
        sys_lines.append(f"**Precision:** `{sys_['precision']}`")
    if sys_.get("checkpoint_url"):
        sys_lines.append(f"**Checkpoint:** <{sys_['checkpoint_url']}>")
    if sys_.get("notes"):
        sys_lines.append(f"**Notes:** {sys_['notes']}")
    out.append(_md_list(sys_lines))
    out.append("")

    # Intended use
    iu = card.get("intended_use", {})
    out.append("## Intended use")
    out.append("**In-scope:**")
    out.append(_md_list(iu.get("in_scope", [])))
    out.append("")
    out.append("**Out-of-scope:**")
    out.append(_md_list(iu.get("out_of_scope", [])))
    out.append("")

    # Fairness scope
    fs = card.get("fairness_scope", {})
    out.append("## Fairness scope")
    for k, label in [
        ("representational_harms", "Representational harms"),
        ("access_harms", "Access harms"),
        ("allocative_harms", "Allocative harms"),
        ("excluded", "Excluded"),
    ]:
        if fs.get(k):
            out.append(f"**{label}:**")
            out.append(_md_list(fs[k]))
            out.append("")

    # Protected attributes
    pa = card.get("protected_attributes", {})
    slices = pa.get("slices", [])
    out.append("## Protected attributes")
    slice_lines = [", ".join(f"`{k}={v}`" for k, v in s.items()) for s in slices]
    out.append(f"**Slices ({len(slices)}):**")
    out.append(_md_list(slice_lines))
    out.append("")
    out.append(f"**Operationalisation:** {pa.get('operationalisation', '?')}")
    out.append("")

    # Prompt suite
    ps = card.get("prompt_suite")
    if ps:
        out.append("## Prompt suite")
        if ps.get("suite_name"):
            out.append(f"**Suite name:** `{ps['suite_name']}`")
        for f in ps.get("families", []):
            out.append("")
            out.append(f"### {f['id']} — {f.get('name', '')}")
            if f.get("description"):
                out.append(f"_{f['description']}_")
            out.append("")
            out.append(f"Canonical template: `{f.get('template', '')}`")
            paras = f.get("paraphrases", [])
            if paras:
                out.append("")
                out.append(f"Paraphrases ({len(paras)}):")
                out.append(_md_list([f"`{p}`" for p in paras]))
        out.append("")
        out.append(f"**Occupations:** {', '.join(ps.get('occupations', []))}")
        out.append(f"**Total prompts:** {ps.get('total_prompts', '?')}")
        out.append(f"**Attribute insertion rule:** {ps.get('attribute_insertion_rule', '?')}")
        out.append("")

    # Decoding
    dec = card.get("decoding")
    if dec:
        out.append("## Decoding")
        out.append("| Regime | Temperature | top_p | max_new_tokens |")
        out.append("|---|---|---|---|")
        for r in dec.get("regimes", []):
            out.append(f"| {r.get('name', '?')} | {r.get('temperature', '?')} | {r.get('top_p', '?')} | {r.get('max_new_tokens', '?')} |")
        out.append("")
        if dec.get("seeds_per_prompt") is not None:
            out.append(f"**Seeds per prompt:** {dec['seeds_per_prompt']}")
        if dec.get("post_processing"):
            out.append(f"**Post-processing:** {dec['post_processing']}")
        out.append("")

    # Refusal handling
    rh = card.get("refusal_handling")
    if rh:
        out.append("## Refusal / deflection accounting")
        out.append(f"**Refusal definition:** {rh.get('refusal_definition', '?')}")
        out.append("")
        if rh.get("deflection_definition"):
            out.append(f"**Deflection definition:** {rh['deflection_definition']}")
            out.append("")
        out.append(f"**Retention policy:** `{rh.get('retention_policy', '?')}`")
        out.append("")

    # Scorer
    sc = card.get("scorer")
    if sc:
        out.append("## Scorer")
        out.append(f"**Type:** `{sc.get('type', '?')}`")
        if sc.get("patterns_file"):
            out.append(f"**Patterns file:** `{sc['patterns_file']}`")
        if sc.get("judge_model"):
            out.append(f"**Judge model:** {sc['judge_model']}")
        if sc.get("annotator_pool"):
            out.append(f"**Annotator pool:** {sc['annotator_pool']}")
        if sc.get("thresholds"):
            ths = ", ".join(f"`{k}={v}`" for k, v in sc["thresholds"].items())
            out.append(f"**Thresholds:** {ths}")
        out.append("")

    # Metrics
    metrics = card.get("metrics", [])
    if metrics:
        out.append("## Metrics")
        for m in metrics:
            unit = f" _({m['unit']})_" if m.get("unit") else ""
            out.append(f"- **{m['name']}**{unit}: {m.get('definition', '')}")
        out.append("")

    # Decision rules
    rules = card.get("decision_rules", [])
    if rules:
        out.append("## Decision rules")
        for r in rules:
            out.append(f"- **Rule:** {r.get('rule', '')}")
            if r.get("rationale"):
                out.append(f"  _Why:_ {r['rationale']}")
        out.append("")

    # Example result
    er = card.get("example_result")
    if er and er.get("rows"):
        out.append("## Example result")
        cols = er.get("columns", [])
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in er["rows"]:
            out.append("| " + " | ".join(str(c) for c in row) + " |")
        out.append("")

    # Reproducibility
    rep = card.get("reproducibility", {})
    if rep:
        out.append("## Reproducibility")
        rep_lines = []
        if rep.get("code_url"):
            rep_lines.append(f"**Code:** <{rep['code_url']}>")
        if rep.get("license"):
            rep_lines.append(f"**License:** {rep['license']}")
        if rep.get("data_release"):
            rep_lines.append(f"**Data release:** `{rep['data_release']}`")
        if rep.get("seeds"):
            rep_lines.append(f"**Seeds:** {rep['seeds']}")
        if rep.get("total_samples") is not None:
            rep_lines.append(f"**Total samples:** {rep['total_samples']}")
        if rep.get("regen_command"):
            rep_lines.append(f"**Regenerate:** `{rep['regen_command']}`")
        out.append(_md_list(rep_lines))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# ──────────────────────────── CLI ────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("card", type=Path, help="Path to a Fairness Card YAML.")
    p.add_argument("--format", choices=["html", "markdown"], default="markdown")
    p.add_argument("--no-lint", action="store_true", help="Skip the lint pass and render whatever's there.")
    args = p.parse_args()

    if not args.card.exists():
        sys.exit(f"file not found: {args.card}")

    with args.card.open() as f:
        card = yaml.safe_load(f)

    if not args.no_lint:
        problems = lint(card)
        if problems:
            print(f"# lint: {len(problems)} issue(s) with {args.card}", file=sys.stderr)
            for msg in problems:
                print(f"#   - {msg}", file=sys.stderr)
            print("# (rendering anyway; pass --no-lint to suppress this banner)", file=sys.stderr)

    if args.format == "html":
        print(render_html(card))
    else:
        print(render_markdown(card))


if __name__ == "__main__":
    main()
