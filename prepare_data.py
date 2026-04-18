#!/usr/bin/env python3
"""
Dataset preparation for Gemma 4 E2B text-to-SQL fine-tuning on Mac.
Produces train.jsonl and valid.jsonl in mlx-lm format.

Key Gemma 4 change: prompt format uses apply_chat_template()
with standard system/user/assistant roles. No <start_of_turn> tokens.
"""

import json, sqlite3, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/gemma-4-E2B-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── System prompt — NO <|think|> token; thinking mode must be OFF ─────────────

SYSTEM = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate a single, correct SQL query. "
    "Output ONLY the SQL query — no explanation, no markdown, no commentary."
)

def make_example(schema: str, question: str, sql: str) -> dict:
    """
    mlx-lm expects {"text": "<full formatted example>"}.
    Uses apply_chat_template for correct Gemma 4 formatting.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Database schema:\n{schema.strip()}\n\n"
                f"Question: {question.strip()}\n\n"
                f"Generate the SQL query:"
            )
        },
        {
            "role": "assistant",
            "content": sql.strip().rstrip(";") + ";"
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# ── SQL execution filter ──────────────────────────────────────────────────────

def is_valid_sql(schema_sql: str, sql: str) -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        if schema_sql:
            conn.executescript(schema_sql)
        conn.execute(sql)
        conn.close()
        return True
    except Exception:
        return False

# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_gretel() -> list[dict]:
    print("Loading Gretel synthetic dataset...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    print(f"Filtering {len(ds)} Gretel samples by SQL executability...")
    valid = []

    def check(row):
        return is_valid_sql(row.get("sql_context", ""), row.get("sql", ""))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(check, row): row for row in ds}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Gretel filter"):
            row = futures[future]
            if future.result():
                valid.append(make_example(
                    schema=row.get("sql_context", ""),
                    question=row.get("sql_prompt", ""),
                    sql=row.get("sql", ""),
                ))
    print(f"  Kept {len(valid)} / {len(ds)} Gretel samples")
    return valid

def load_spider_train() -> list[dict]:
    print("Loading Spider training set...")
    ds = load_dataset("spider", split="train")
    examples = []
    for row in ds:
        schema = f"-- Database: {row.get('db_id', 'db')}"
        examples.append(make_example(
            schema=schema,
            question=row["question"],
            sql=row["query"],
        ))
    print(f"  Loaded {len(examples)} Spider samples")
    return examples

def load_bird_train(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"   BIRD not found at {path} — download from https://bird-bench.github.io/")
        return []
    with open(p) as f:
        data = json.load(f)
    examples = []
    for row in data:
        schema = row.get("evidence", "") + f"\n-- Database: {row.get('db_id', 'db')}"
        examples.append(make_example(
            schema=schema,
            question=row["question"],
            sql=row["SQL"],
        ))
    print(f"  Loaded {len(examples)} BIRD samples")
    return examples

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    gretel = load_gretel()
    spider = load_spider_train()
    bird   = load_bird_train("./bird/train/train.json")

    # Weight human-annotated data 2× by repeating
    all_examples = gretel + spider + spider + bird + bird
    random.shuffle(all_examples)

    split_idx  = int(len(all_examples) * 0.95)
    train_set  = all_examples[:split_idx]
    valid_set  = all_examples[split_idx:]

    def write_jsonl(data, path):
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    write_jsonl(train_set, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid_set, OUTPUT_DIR / "valid.jsonl")
    write_jsonl(valid_set[:200], OUTPUT_DIR / "test.jsonl")

    print(f"\n Done!")
    print(f"   Train : {len(train_set):,} → data/train.jsonl")
    print(f"   Valid : {len(valid_set):,} → data/valid.jsonl")

if __name__ == "__main__":
    main()
