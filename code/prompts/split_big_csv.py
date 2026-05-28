#!/usr/bin/env python3
import math
from pathlib import Path
import pandas as pd

INPUT = Path("prompt_expanded_big.csv")
OUTDIR = Path("big_chunks")
CHUNK_SIZE = 400

OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)
n = len(df)
num_chunks = math.ceil(n / CHUNK_SIZE)

for i in range(num_chunks):
    start = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, n)
    chunk = df.iloc[start:end].copy()
    chunk.to_csv(OUTDIR / f"prompt_chunk_{i:02d}.csv", index=False)

print(f"Wrote {num_chunks} chunks to {OUTDIR}")
