
import json
import sqlite3
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path("data/spider_data")

records = json.loads(
    Path("results/spider_e18_lora.json").read_text()
)

failures = [
    r for r in records
    if not r["passed"] and r["error"]
][:10]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.float16,
    device_map="cuda",
)

model = PeftModel.from_pretrained(
    base, "adapters/sql_lora_v4"
)
model.eval()


def generate(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


results = []

for i, record in enumerate(failures, 1):
    db_id = record["db_id"]
    path = ROOT / "database" / db_id / f"{db_id}.sqlite"

    conn = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro",
        uri=True,
    )
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA trusted_schema = OFF")

    schema = "\n".join(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL"
        )
    )

    def readonly(action, a1, a2, database, source):
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
        }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

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
        {"role": "user", "content": record["question"]},
        {"role": "assistant", "content": record["generated_sql"]},
        {
            "role": "user",
            "content": (
                f"SQLite execution error: {record['error']}\n"
                "Correct the SQL using the schema. "
                "Return only one corrected SELECT query."
            ),
        },
    ]

    corrected_sql = generate(messages)

    try:
        expected = conn.execute(record["reference_sql"]).fetchall()
        actual = conn.execute(corrected_sql).fetchall()

        # Compare unordered results while preserving duplicates.
        from collections import Counter

        def normalized(rows):
            return Counter(tuple(row) for row in rows)

        if "ORDER BY" in record["reference_sql"].upper():
            passed = actual == expected
        else:
            passed = normalized(actual) == normalized(expected)

        error = None

    except sqlite3.Error as exc:
        actual = None
        passed = False
        error = str(exc)

    results.append({
        "db_id": db_id,
        "question": record["question"],
        "initial_sql": record["generated_sql"],
        "initial_error": record["error"],
        "corrected_sql": corrected_sql,
        "actual": actual,
        "passed": passed,
        "error": error,
    })

    print(
        f"[{i}/{len(failures)}] "
        f"{'RECOVERED' if passed else 'FAILED'} "
        f"| {db_id}",
        flush=True,
    )

    if not passed:
        print("  Corrected:", corrected_sql)
        print("  Error:", error)

    conn.close()

Path("results").mkdir(exist_ok=True)

Path("results/recovery_e20.json").write_text(
    json.dumps(results, indent=2)
)

recovered = sum(r["passed"] for r in results)

print("\n=== E20 RECOVERY SUMMARY ===")
print("Initial correct: 0")
print(f"Recovered: {recovered}/{len(results)}")
print(f"Recovery rate: {recovered / len(results):.1%}")
print("Saved: results/recovery_e20.json")
