
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# E25 — Collect Student Failures
# ============================================================

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "adapters/sql_lora_v4"

DATASET_PATH = Path("data/distill/train.json")
DATABASE_ROOT = Path("data/spider_data/database")

RESULTS_DIR = Path("results")
FAILURES_PATH = Path("data/distill/student_failures.json")

VARIANT = os.environ.get("MODEL_VARIANT", "lora")

if VARIANT not in {"base", "lora"}:
    raise ValueError(f"Invalid MODEL_VARIANT: {VARIANT}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / f"distill_train_{VARIANT}_baseline.json"


# ============================================================
# Load dataset
# ============================================================

examples = json.loads(DATASET_PATH.read_text())

assert len(examples) == 100, (
    f"Expected 100 training questions, got {len(examples)}"
)

print(f"Dataset: {DATASET_PATH}")
print(f"Questions: {len(examples)}")
print(f"Databases: {len(set(x['db_id'] for x in examples))}")
print(f"Model variant: {VARIANT}", flush=True)


# ============================================================
# Load model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda",
)

if VARIANT == "lora":
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

model.eval()


# ============================================================
# Helpers
# ============================================================

def normalize(rows):
    return Counter(
        tuple(
            round(v, 6) if isinstance(v, float) else v
            for v in row
        )
        for row in rows
    )


def readonly(action, arg1, arg2, dbname, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }

    if action in allowed:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def execute(conn, sql):
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    if sqlite3.complete_statement(sql):
        # Reject multiple statements; sqlite3.execute also enforces this.
        pass

    return conn.execute(sql).fetchall()


def get_schema(conn):
    return "\n".join(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL"
        )
    )


def generate_sql(question, schema):
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
        {
            "role": "user",
            "content": question,
        },
    ]

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
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


# ============================================================
# Evaluation
# ============================================================

results = []

for i, example in enumerate(examples, 1):
    db_id = example["db_id"]
    question = example["question"]
    reference_sql = example["reference_sql"]

    db_path = (
        DATABASE_ROOT
        / db_id
        / f"{db_id}.sqlite"
    )

    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(
        f"file:{db_path.resolve()}?mode=ro",
        uri=True,
    )

    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA trusted_schema = OFF")

    schema = get_schema(conn)
    conn.set_authorizer(readonly)

    generated_sql = generate_sql(question, schema)

    expected = None
    actual = None

    reference_error = None
    student_error = None

    # Evaluate reference SQL independently.
    try:
        expected = execute(conn, reference_sql)

    except (sqlite3.Error, ValueError) as exc:
        reference_error = str(exc)

    # Evaluate student SQL independently.
    try:
        actual = execute(conn, generated_sql)

    except (sqlite3.Error, ValueError) as exc:
        student_error = str(exc)

    passed = False

    # Score only if both queries executed successfully.
    if reference_error is None and student_error is None:
        if re.search(r"\bORDER\s+BY\b", reference_sql, re.I):
            passed = expected == actual
        else:
            passed = normalize(expected) == normalize(actual)

    # Keep the legacy error field for compatibility with
    # earlier recovery scripts.
    error = student_error or reference_error

    result = {
        "db_id": db_id,
        "question": question,
        "reference_sql": reference_sql,
        "generated_sql": generated_sql,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "error": error,
        "reference_error": reference_error,
        "student_error": student_error,
    }

    results.append(result)

    status = (
        "REFERENCE ERROR" if reference_error
        else "PASS" if passed
        else "FAIL"
    )

    print(
        f"[{i:03}/{len(examples)}] "
        f"{db_id}: {status}",
        flush=True,
    )

    if student_error:
        print(f"  Student error: {student_error}")

    if reference_error:
        print(f"  Reference error: {reference_error}")

    conn.close()


# ============================================================
# Save results
# ============================================================

OUTPUT_PATH.write_text(
    json.dumps(results, indent=2)
)

# Only collect student failures where the reference query
# executed successfully.
failures = [
    r for r in results
    if not r["passed"] and r["reference_error"] is None
]

if VARIANT == "lora":
    FAILURES_PATH.write_text(
        json.dumps(failures, indent=2)
    )


# ============================================================
# Summary
# ============================================================

correct = sum(r["passed"] for r in results)

reference_errors = sum(
    r["reference_error"] is not None for r in results
)

student_errors = sum(
    r["student_error"] is not None for r in results
)

print("\n=== E25 BASELINE SUMMARY ===")

print(f"Questions: {len(results)}")
print(f"Correct: {correct}")
print(f"Incorrect: {len(results) - correct}")
print(f"Accuracy: {correct / len(results):.1%}")

print(f"Reference execution errors: {reference_errors}")
print(f"Student execution errors: {student_errors}")

print(f"Usable student failures: {len(failures)}")

print(f"\nSaved: {OUTPUT_PATH}")

if VARIANT == "lora":
    print(f"Failures saved: {FAILURES_PATH}")
