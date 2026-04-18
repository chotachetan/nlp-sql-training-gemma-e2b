#!/usr/bin/env python3
"""
Build DPO preference pairs using Gemma 4 E2B SFT model + SQL execution feedback.
Uses the 4-bit quantized model for faster candidate generation.
"""

import json, sqlite3, random
from pathlib import Path
from transformers import AutoTokenizer
from mlx_lm import load, generate

BIRD_DEV  = "./bird/dev/dev.json"
OUT_PATH  = Path("./data/dpo_pairs.jsonl")
SFT_MODEL = str(Path.home() / "models/gemma-4-e2b-it-mlx-4bit")
# Use 4-bit quantized version for speed during candidate generation
N_SAMPLES    = 1500
N_CANDIDATES = 4

SYSTEM = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate a single, correct SQL query. "
    "Output ONLY the SQL query — no explanation, no markdown, no commentary."
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

def make_inference_prompt(schema: str, question: str) -> str:
    """Prompt without assistant turn — for inference."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Database schema:\n{schema.strip()}\n\n"
            f"Question: {question.strip()}\n\nGenerate the SQL query:"
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def execute_sql(sql: str, schema: str):
    try:
        conn = sqlite3.connect(":memory:")
        if schema:
            conn.executescript(schema)
        result = conn.execute(sql).fetchall()
        conn.close()
        return True, result
    except Exception:
        return False, None

def extract_sql(raw: str) -> str:
    """Strip any trailing special tokens from generated output."""
    for stop in ["<|im_end|>", "</s>", "<eos>"]:
        raw = raw.split(stop)[0]
    return raw.strip()

def main():
    if not Path(BIRD_DEV).exists():
        print(f"BIRD dev set not found at {BIRD_DEV}")
        print("Download from https://bird-bench.github.io/")
        return

    print(f"Loading SFT model from {SFT_MODEL}...")
    model, _ = load(SFT_MODEL)

    with open(BIRD_DEV) as f:
        bird_dev = json.load(f)
    random.seed(42)
    random.shuffle(bird_dev)
    samples = bird_dev[:N_SAMPLES]

    dpo_pairs = []
    for i, row in enumerate(samples):
        schema   = row.get("evidence", "") + f"\n-- Database: {row.get('db_id', '')}"
        question = row["question"]
        ref_sql  = row["SQL"]
        prompt   = make_inference_prompt(schema, question)

        candidates = []
        for _ in range(N_CANDIDATES):
            out = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=200,
                temp=0.8,
                top_p=0.95,
                verbose=False,
            )
            candidates.append(extract_sql(out))

        _, ref_result = execute_sql(ref_sql, schema)
        winners, losers = [], []
        for sql in candidates:
            ok, result = execute_sql(sql, schema)
            if ok and result == ref_result:
                winners.append(sql)
            else:
                losers.append(sql)

        for w in winners[:1]:
            for l in losers[:1]:
                dpo_pairs.append({
                    "prompt":   prompt,
                    "chosen":   w,
                    "rejected": l,
                })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] Pairs so far: {len(dpo_pairs)}")

    print(f"\nTotal DPO pairs: {len(dpo_pairs)}")
    with open(OUT_PATH, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
