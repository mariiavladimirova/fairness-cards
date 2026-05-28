#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM

import traceback 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Local path to model directory, e.g. $WORK/fairness_ablation/models/Qwen2.5-7B-Instruct")
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV with prompts and generation params")
    p.add_argument("--output_jsonl", type=str, required=True,
                   help="Path to JSONL output file")
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_rows", type=int, default=None,
                   help="Optional limit for smoke tests")
    p.add_argument("--attn_implementation", type=str, default=None,
                   help="Optional, e.g. flash_attention_2 if supported")
    return p.parse_args()


def get_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def ensure_offline_env():
    # Make Hugging Face / Transformers fail fast if something tries network access
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_completed_ids(output_path: Path):
    completed = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add(str(rec["row_id"]))
            except Exception:
                # Ignore malformed lines rather than crashing resume logic
                continue
    return completed


def append_jsonl(output_path: Path, record: dict):
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def generate_one(model, tokenizer, system_prompt, user_prompt, temperature, top_p, seed, max_new_tokens):
    prompt_text = build_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    # Put inputs on the same device as the first input embedding weights
    target_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    # Simpler and more robust than torch.Generator(device=...)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[prompt_len:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if not output_text:
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        prompt_text_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True).strip()
        if full_text.startswith(prompt_text_decoded):
            output_text = full_text[len(prompt_text_decoded):].strip()
        else:
            output_text = full_text

    return output_text


def main():
    args = parse_args()
    ensure_offline_env()

    model_path = Path(args.model_path).expanduser().resolve()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    print(f"Loading prompts from: {input_csv}")
    df = pd.read_csv(input_csv, dtype={"row_id": str})

    required_cols = [
        "row_id", "prompt_text", "temperature", "top_p", "seed"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if "max_new_tokens" not in df.columns:
        df["max_new_tokens"] = 128

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    completed = load_completed_ids(output_jsonl)
    if completed:
        print(f"Found {len(completed)} completed rows in {output_jsonl}; skipping them.")
        df = df[~df["row_id"].astype(str).isin(completed)].copy()

    print(f"Rows remaining to run: {len(df)}")
    if len(df) == 0:
        print("Nothing to do.")
        return

    dtype = get_dtype(args.dtype)

    print(f"Loading model from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        **model_kwargs,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    for row in tqdm(df.to_dict("records"), total=len(df)):
        row_id = str(row["row_id"])
        start = time.time()
        status = "ok"
        error = None
        output_text = None

        try:
            output_text = generate_one(
                model=model,
                tokenizer=tokenizer,
                system_prompt=args.system_prompt,
                user_prompt=row["prompt_text"],
                temperature=row["temperature"],
                top_p=row["top_p"],
                seed=row["seed"],
                max_new_tokens=row.get("max_new_tokens", 128),
            )
        except Exception as e:
            status = "error"
            error = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        elapsed = time.time() - start

        record = {
            "row_id": row_id,
            "run_started_at": run_started_at,
            "model_path": str(model_path),
            "prompt_family": row.get("prompt_family"),
            "paraphrase_id": row.get("paraphrase_id"),
            "occupation": row.get("occupation"),
            "slice_gender": row.get("slice_gender"),
            "slice_religion": row.get("slice_religion"),
            "full_slice_id": row.get("full_slice_id"),
            "prompt_text": row.get("prompt_text"),
            "temperature": row.get("temperature"),
            "top_p": row.get("top_p"),
            "seed": row.get("seed"),
            "max_new_tokens": row.get("max_new_tokens", 128),
            "status": status,
            "error": error,
            "elapsed_sec": elapsed,
            "output_text": output_text,
        }
        append_jsonl(output_jsonl, record)

    print(f"Done. Results written to: {output_jsonl}")


if __name__ == "__main__":
    main()
