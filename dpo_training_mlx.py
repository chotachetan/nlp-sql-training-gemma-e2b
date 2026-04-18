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
