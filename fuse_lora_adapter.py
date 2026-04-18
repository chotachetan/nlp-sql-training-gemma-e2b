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
