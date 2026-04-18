# Training Gemma 4 E2B for NLP-to-SQL on MacBook Pro M4 Max 64GB
### A Mac-native guide using Apple's MLX framework.


## Why MLX — Not PyTorch — on Apple Silicon

MLX was designed from scratch for Apple Silicon's unified memory. CPU, GPU,
and the Neural Engine share the same physical memory pool — no copies.
PyTorch's MPS backend still copies tensors between pools.

On a 64 GB M4 Max with Gemma 4 E2B:

| What | Cloud A100 (QLoRA) | M4 Max 64GB (MLX LoRA) |
|---|---|---|
| Gemma 4 E2B full LoRA | Tight at 40GB | ~12 GB used |
| Gemma 4 E2B QLoRA (4-bit) | Fits easily | ~6 GB used |
| Training speed | ~700–900 tok/sec | ~350–550 tok/sec |
| DPO inference (candidate gen) | Fast | ~180–250 tok/sec |
| Cost | ~$2–4/hr | $0 |

With 64 GB unified memory you can run full LoRA (no quantization needed),
giving better adapter quality than QLoRA.

---

## Overview of Steps

```
Step 1   Install system tools (Homebrew, Python, Xcode CLT)
Step 2   Set up Python environment + MLX
Step 3   Authenticate with Hugging Face
Step 4   Download & convert Gemma 4 E2B to MLX format
Step 5   Prepare datasets
Step 6   Stage 1 — SFT with mlx_lm LoRA
Step 7   Fuse LoRA adapters into the base model
Step 8   Stage 2 — DPO with mlx-tune
Step 9   Evaluate on Spider + BIRD
Step 10  Export to GGUF and bundle with Ollama
```

Estimated total wall-clock time on M4 Max 64GB: **10–15 hours**
(Gemma 4 E2B's PLE architecture adds ~20% overhead vs Gemma 2 2B).

---

## Step 1 — Install System Tools

```bash
# Xcode Command Line Tools (required for compilation)
xcode-select --install

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11
brew install python@3.11

# git-lfs (required for HuggingFace model downloads)
brew install git-lfs
git lfs install

# Ollama (for serving after training)
brew install ollama

# llama.cpp (for GGUF export at the end)
brew install llama.cpp

# Verify Metal GPU is visible
python3.11 -c "
import subprocess
out = subprocess.run(['system_profiler','SPDisplaysDataType'],
                    capture_output=True, text=True).stdout
print([l for l in out.splitlines() if 'Chipset' in l or 'Metal' in l])"
```

---

## Step 2 — Set Up Python Environment

```bash
python3.11 -m venv ~/mlx-text2sql
source ~/mlx-text2sql/bin/activate

pip install --upgrade pip

# Core MLX stack
pip install mlx
pip install mlx-lm              # Fine-tuning + inference
pip install mlx-tune            # DPO support on Mac (Unsloth-compatible API)

# Transformers — only for tokenizer and apply_chat_template
# Must be >=4.51.0 for Gemma 4 support
pip install "transformers>=4.51.0"

# Data + evaluation
pip install datasets huggingface_hub
pip install sqlglot sqlparse
pip install pandas tqdm
pip install wandb   # Optional experiment tracking

# Verify MLX GPU
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

---

## Step 3 — Authenticate with Hugging Face

Gemma 4 uses Apache 2.0 — no commercial restrictions — but you still need
to accept Google's terms on HuggingFace before downloading.

```bash
# 1. Go to https://huggingface.co/google/gemma-4-E2B-it
# 2. Click "Agree and access repository"

# 3. Log in via CLI
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

---

## Step 4 — Download and Convert Gemma 4 E2B to MLX Format

MLX uses its own optimized array format. This conversion is a one-time step.

```bash
# Full bfloat16 — no quantization needed with 64GB unified memory
# Gemma 4 E2B is ~10–12 GB in bf16 due to PLE embeddings
python -m mlx_lm.convert \
    --hf-path google/gemma-4-E2B-it \
    --mlx-path ~/models/gemma-4-e2b-it-mlx-bf16

# Takes ~15–25 minutes depending on internet speed.

# Also create a 4-bit quantized version for fast DPO candidate generation
python -m mlx_lm.convert \
    --hf-path google/gemma-4-E2B-it \
    --mlx-path ~/models/gemma-4-e2b-it-mlx-4bit \
    --quantize \
    --q-bits 4

# Quick sanity check — verify the model generates text
python -m mlx_lm.generate \
    --model ~/models/gemma-4-e2b-it-mlx-bf16 \
    --prompt "SELECT employees FROM" \
    --max-tokens 30
```

---

## Step 5 — Prepare the Training Data

SQLite is built into macOS, so execution-based SQL filtering works natively.

### Step 5.1 — Create the data preparation script

Save as `~/mlx-text2sql/prepare_data.py`:

```python
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
```

```bash
cd ~/mlx-text2sql
python prepare_data.py
# Takes ~25–45 minutes (mostly the Gretel filter pass)
# Expected: ~115k–120k total training examples
```

---

## Step 6 — Stage 1: SFT with mlx_lm LoRA

### Step 6.1 — Create the training config

Save as `~/mlx-text2sql/sft_config.yaml`:

```yaml
# Gemma 4 E2B SFT config for M4 Max 64GB
# Full LoRA (no quantization) — 64GB unified memory has plenty of headroom

model: ~/models/gemma-4-e2b-it-mlx-bf16

train: true
data: ./data

# LoRA settings
# lora_layers=18 covers all transformer layers in E2B
lora_layers: 18

# Training hyperparameters
batch_size: 6          # 64GB allows batch 6 at bf16 for Gemma 4 E2B
                       # (slightly less than Gemma 2 2B due to larger PLE embeddings)
iters: 20000
val_batches: 50
steps_per_report: 100
steps_per_eval: 500
save_every: 1000
adapter_path: ./adapters/sft

# Optimizer
learning_rate: 2e-4
lr_schedule:
  name: cosine_decay
  warmup: 200

max_seq_length: 1024

# Gradient checkpointing — optional at 64GB, but good practice
grad_checkpoint: false
```

### Step 6.2 — Launch training

```bash
cd ~/mlx-text2sql
mkdir -p logs adapters/sft

python -m mlx_lm.lora \
    --config sft_config.yaml \
    2>&1 | tee logs/sft_training.log
```

Expected log output:

```
Iter 100: Train loss 2.291, It/sec 3.8, Tokens/sec 387
Iter 200: Train loss 1.843, It/sec 3.9, Tokens/sec 398
Iter 500: Val loss 1.176
...
```

**Expected training time on M4 Max 64GB:** ~7–10 hours for 20,000 iterations.

For a quick smoke test, set `iters: 2000` first to confirm loss drops cleanly
before committing to the full run.

### Step 6.3 — Monitor during training (separate terminal)

```bash
# Memory usage
memory_pressure

# GPU activity
sudo powermetrics --samplers gpu_power -i 1000 | grep "GPU Active"

# Training log tail
tail -f ~/mlx-text2sql/logs/sft_training.log
```

---

## Step 7 — Fuse LoRA Adapters into the Base Model

```bash
python -m mlx_lm.fuse \
    --model ~/models/gemma-4-e2b-it-mlx-bf16 \
    --adapter-path ./adapters/sft \
    --save-path ~/models/gemma-4-e2b-text2sql-sft

# Verify the fused model generates clean SQL
python - <<'EOF'
from transformers import AutoTokenizer
from mlx_lm import load, generate

MODEL = str(__import__("pathlib").Path.home() / "models/gemma-4-e2b-text2sql-sft")
SYSTEM = "You are an expert SQL assistant. Output ONLY the SQL query."

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model, _ = load(MODEL)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": (
        "Database schema:\nCREATE TABLE employees "
        "(id INT, name TEXT, dept TEXT, salary INT);\n\n"
        "Question: What is the average salary by department?\n\n"
        "Generate the SQL query:"
    )},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
out = generate(model, tokenizer, prompt=prompt, max_tokens=100, temp=0.0, verbose=False)
print("Generated SQL:", out.strip())
# Expected: SELECT dept, AVG(salary) FROM employees GROUP BY dept;
EOF
```

---

## Step 8 — Stage 2: DPO with mlx-tune

### Step 8.1 — Generate DPO preference pairs

Save as `~/mlx-text2sql/build_dpo_data.py`:

```python
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
```

```bash
python build_dpo_data.py
# Takes ~1.5–2.5 hours (generating 1500×4 = 6000 SQL candidates)
# Expected output: 400–900 DPO pairs
```

### Step 8.2 — Run DPO training with mlx-tune

```bash
python - <<'EOF'
from mlx_tune import FastLanguageModel, DPOTrainer
from datasets import Dataset
import json
from pathlib import Path

SFT_MODEL = str(Path.home() / "models/gemma-4-e2b-text2sql-sft")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_MODEL,
    max_seq_length=1024,
    load_in_4bit=False,     # Full bf16 on 64GB
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

pairs = [json.loads(l) for l in open("./data/dpo_pairs.jsonl")]
dpo_dataset = Dataset.from_list(pairs)

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dpo_dataset,
    beta=0.1,
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    max_length=1024,
    output_dir="./adapters/dpo",
)

trainer.train()
trainer.save_model("./adapters/dpo")
print("DPO training complete.")
EOF
```

### Step 8.3 — Fuse DPO adapters into SFT model

```bash
python -m mlx_lm.fuse \
    --model ~/models/gemma-4-e2b-text2sql-sft \
    --adapter-path ./adapters/dpo \
    --save-path ~/models/gemma-4-e2b-text2sql-FINAL

echo " Final model saved to ~/models/gemma-4-e2b-text2sql-FINAL"
```

---

## Step 9 — Evaluate on Spider + BIRD

Save as `~/mlx-text2sql/evaluate.py`:

```python
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
```

```bash
python evaluate.py 2>&1 | tee logs/eval_results.log
# Takes ~15–25 minutes at ~200 tok/sec inference
```

---

## Step 10 — Export to GGUF and Bundle with Ollama

### Step 10.1 — Verify the fused model is in HF format

```bash
ls ~/models/gemma-4-e2b-text2sql-FINAL/
# Should show: config.json, tokenizer.json, model-*.safetensors, etc.
```

### Step 10.2 — Convert to GGUF

```bash
# llama.cpp was installed in Step 1 via Homebrew

# Convert HF safetensors → GGUF F16
llama-convert-hf-to-gguf \
    ~/models/gemma-4-e2b-text2sql-FINAL \
    --outtype f16 \
    --outfile ~/models/gemma-4-e2b-text2sql-f16.gguf

# Quantize to Q4_K_M (~2.5 GB)
llama-quantize \
    ~/models/gemma-4-e2b-text2sql-f16.gguf \
    ~/models/gemma-4-e2b-text2sql-q4km.gguf \
    Q4_K_M

echo "Model size: $(du -sh ~/models/gemma-4-e2b-text2sql-q4km.gguf)"
# Expected: ~2.5G
```

### Step 10.3 — Create Ollama Modelfile and register

Gemma 4's stop token is `<|im_end|>`, not `<end_of_turn>`:

```bash
cat > ~/models/Modelfile << 'EOF'
FROM /Users/YOUR_USERNAME/models/gemma-4-e2b-text2sql-q4km.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are an expert SQL assistant. Given a database schema and a natural language question, generate a single, correct SQL query. Output ONLY the SQL query with no explanation, markdown, or commentary."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 2048
EOF

sed -i '' "s/YOUR_USERNAME/$USER/g" ~/models/Modelfile

ollama create gemma4-e2b-text2sql -f ~/models/Modelfile

# Test
ollama run gemma4-e2b-text2sql \
  "Database schema:
CREATE TABLE sales (id INT, product TEXT, region TEXT, amount FLOAT, date DATE);

Question: What is the total sales amount by region for Q1 2024?"
```

---

## Time & Memory Estimates (M4 Max 64GB)

| Stage | Time | Peak RAM |
|---|---|---|
| Model download + MLX conversion | ~20–30 min | ~12 GB |
| Data preparation (Gretel filter) | ~30–45 min | ~8 GB |
| Stage 1 SFT — 20k iters, batch 6 | ~7–10 hrs | ~24–32 GB |
| Adapter fusion | ~5–8 min | ~14 GB |
| DPO candidate generation (1500 samples) | ~1.5–2.5 hrs | ~8 GB (4-bit) |
| Stage 2 DPO — 1 epoch | ~35–55 min | ~16 GB |
| Evaluation (600 samples) | ~20–25 min | ~10 GB |
| GGUF conversion + quantization | ~12 min | ~12 GB |
| **Total** | **~11–15 hrs** | **~32 GB peak** |

> Peak RAM is ~32 GB during SFT — well within your 64 GB. You can keep
> a browser, Activity Monitor, and VS Code open throughout.

---

## Apple Silicon Tips

**Keep plugged in — disable Low Power Mode.**
System Settings → Battery → Low Power Mode → Never.
Power saver throttles the GPU and adds 30–50% to training time.

**Prevent sleep during training.**
```bash
caffeinate -t 54000 &   # 15 hours
```

**Monitor GPU utilization.**
```bash
sudo powermetrics --samplers gpu_power -i 1000 | grep "GPU Active"
```

**If mlx_lm throws a stale adapter error**, delete the adapter directory
and restart:
```bash
rm -rf ./adapters/sft && python -m mlx_lm.lora --config sft_config.yaml
```

**Check for thermal throttling** during long runs:
```bash
sudo powermetrics --samplers thermal -i 3000 | grep "CPU die temperature"
# Normal under load: 65–80°C. Throttling begins above ~95°C.
```

**Gemma 4 E2B is ~20% slower than Gemma 2 2B on MLX** due to its
Per-Layer Embedding (PLE) architecture, which adds a larger lookup table
per layer. This is expected — not a configuration error.
