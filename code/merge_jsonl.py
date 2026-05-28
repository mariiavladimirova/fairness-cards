#!/usr/bin/env python3
from pathlib import Path

input_dir = Path("runs")
output_file = Path("runs/big_study_merged.jsonl")

files = sorted(input_dir.glob("prompt_chunk_*.jsonl"))

with output_file.open("w", encoding="utf-8") as out:
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)

print(f"Merged {len(files)} files into {output_file}")
