
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
import os
from peft import PeftModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = Path("data/spider_data")
DATABASES = [
    "concert_singer",
    "pets_1",
    "car_1",
    "flight_2",
    "employee_hire_evaluation",
]


examples = json.loads((DATA / "dev.json").read_text())
selected = json.loads(
    Path("data/spider_eval_100.json").read_text()
)

def normalize(rows):
    return Counter(
        tuple(round(v, 6) if isinstance(v, float) else v for v in row)
        for row in rows
    )

def readonly(action, arg1, arg2, dbname, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

def execute(conn, sql):
    # Controlled development benchmark only, not a secure SQL sandbox.
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Expected one SELECT query")
    return conn.execute(sql).fetchall()

print(f"Evaluating {len(selected)} questions across {len(DATABASES)} databases")

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16, device_map="cuda"
)

variant = os.environ.get("MODEL_VARIANT", "base")

if variant not in ("base", "lora"):
    raise ValueError(f"Unknown model variant: {variant}")

if variant == "lora":
    model = PeftModel.from_pretrained(
        model,
        "adapters/sql_lora_v4",
    )

model.eval()

print("Evaluating variant:", variant, flush=True)
model.eval()

results = []

for i, ex in enumerate(selected, 1):
    db_id = ex["db_id"]
    db_path = DATA / "database" / db_id / f"{db_id}.sqlite"

    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA trusted_schema = OFF")

    schema = "\n".join(
        row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL"
        )
    )
    conn.set_authorizer(readonly)

    messages = [
        {
            "role": "system",
            "content": (
                "Translate the user's question into SQLite SQL. "
                "Return exactly one SELECT statement and no explanation. "
                "Return only the columns needed to answer the question.\n\n"
                f"Database schema:\n{schema}"
            ),
        },
        {"role": "user", "content": ex["question"]},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    if db_id == "flight_2" and i == 19:
     print("\n=== INPUT DIAGNOSTIC ===")
     print("Schema characters:", len(schema))
     print("Prompt tokens:", inputs.shape[-1])
     print("Model context limit:", model.config.max_position_embeddings)

    with torch.inference_mode():
        output = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    sql = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    expected = actual = None
    error = None
    passed = False

    try:
        expected = execute(conn, ex["query"])
        actual = execute(conn, sql)

        if re.search(r"\bORDER\s+BY\b", ex["query"], re.I):
            passed = expected == actual
        else:
            passed = normalize(expected) == normalize(actual)

    except (sqlite3.Error, ValueError) as exc:
        error = str(exc)

    conn.close()

    results.append({
        "db_id": db_id,
        "question": ex["question"],
        "reference_sql": ex["query"],
        "generated_sql": sql,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "error": error,
    })

    print(
        f"[{i:02}/{len(selected)}] {db_id}: "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )

    if not passed:
        print("  Question:", ex["question"])
        print("  Generated:", sql)
        print("  Error:", error)
        print("  Expected:", expected)
        print("  Actual:", actual)

Path("results").mkdir(exist_ok=True)
out = Path(f"results/spider_e18_{variant}.json")
out.write_text(json.dumps(results, indent=2))

correct = sum(r["passed"] for r in results)

print("\n=== E5 SUMMARY ===")
print(f"Passed: {correct}/{len(results)}")
print(f"Success rate: {correct / len(results):.1%}")

print("\nPer database:")
for db in DATABASES:
    subset = [r for r in results if r["db_id"] == db]
    n = sum(r["passed"] for r in subset)
    print(f"  {db}: {n}/{len(subset)}")

print(f"\nSaved: {out}")
