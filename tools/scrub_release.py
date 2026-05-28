"""Scrub identifying paths from released data files.

Replaces absolute `model_path` strings in the merged JSONL with the basename
of the checkpoint. Operates in-place on a target file (writes to .tmp first
and renames atomically).

Usage:
    python tools/scrub_release.py runs/big_study_merged.jsonl
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def scrub_record(rec: dict) -> dict:
    mp = rec.get("model_path")
    if isinstance(mp, str) and mp:
        rec["model_path"] = os.path.basename(mp.rstrip("/"))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.target.exists():
        sys.exit(f"file not found: {args.target}")

    tmp = args.target.with_suffix(args.target.suffix + ".tmp")
    n_changed = 0
    with args.target.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            before = rec.get("model_path")
            rec = scrub_record(rec)
            if rec.get("model_path") != before:
                n_changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.dry_run:
        tmp.unlink()
        print(f"[dry-run] would scrub {n_changed} records in {args.target}")
        return

    shutil.move(str(tmp), str(args.target))
    print(f"scrubbed {n_changed} records in {args.target}")


if __name__ == "__main__":
    main()
