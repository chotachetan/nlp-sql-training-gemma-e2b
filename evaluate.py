#!/usr/bin/env python3
"""Evaluation for Gemma 4 E2B text-to-SQL using mlx_lm inference."""

import json, sqlite3
from pathlib import Path
from transformers import AutoTokenizer
from mlx_lm import load, generate
from datasets import load_dataset
import sqlglot

FINAL_MODEL = str(Path.home() / "models/gemma-4-e2b-text2sql-FINAL")
SYSTEM = ("You are an expert SQL assistant. Output ONLY the SQL query.")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
model, _  = load(FINAL_MODEL)

def make_prompt(schema: str, question: str) -> str:
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

def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(
            sql, read="sqlite", write="sqlite", pretty=False
        )[0].lower().strip()
    except Exception:
        return sql.lower().strip()

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
    for stop in ["<|im_end|>", "</s>", "<eos>"]:
        raw = raw.split(stop)[0]
    return raw.strip()

# ── Spider Exact Match ────────────────────────────────────────────────────────
print("\n--- Spider Exact Match ---")
spider_dev = load_dataset("spider", split="validation")
em_correct = em_total = 0

for i, row in enumerate(spider_dev.select(range(300))):
    schema = f"-- Database: {row.get('db_id', 'db')}"
    prompt = make_prompt(schema, row["question"])
    raw    = generate(model, tokenizer, prompt=prompt,
                      max_tokens=200, temp=0.0, verbose=False)
    pred   = extract_sql(raw)

    if normalize_sql(pred) == normalize_sql(row["query"]):
        em_correct += 1
    em_total += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/300] EM: {em_correct/em_total*100:.1f}%")

print(f"\nSpider Exact Match: {em_correct}/{em_total} = {em_correct/em_total*100:.1f}%")

# ── BIRD Execution Accuracy ───────────────────────────────────────────────────
print("\n--- BIRD Execution Accuracy ---")
bird_path = "./bird/dev/dev.json"

if not Path(bird_path).exists():
    print(f"BIRD dev set not found at {bird_path} — skipping.")
else:
    with open(bird_path) as f:
        bird_dev = json.load(f)[:300]

    ex_correct = ex_total = 0
    for i, row in enumerate(bird_dev):
        schema = row.get("evidence", "") + f"\n-- Database: {row.get('db_id', '')}"
        prompt = make_prompt(schema, row["question"])
        raw    = generate(model, tokenizer, prompt=prompt,
                          max_tokens=256, temp=0.0, verbose=False)
        pred   = extract_sql(raw)

        pred_ok, pred_res = execute_sql(pred, schema)
        ref_ok,  ref_res  = execute_sql(row["SQL"], schema)

        if pred_ok and ref_ok and pred_res == ref_res:
            ex_correct += 1
        ex_total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/300] EX: {ex_correct/ex_total*100:.1f}%")

    print(f"\nBIRD Execution Accuracy: {ex_correct}/{ex_total} = {ex_correct/ex_total*100:.1f}%")
