
import gc
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
DB_ROOT = Path("data/spider_data/database")
DATA = Path("data/distill/val.json")
OUT = Path("results/e37_comparison.json")

ADAPTERS = {
    "v4": "adapters/sql_lora_v4",
    "e32": "adapters/sql_lora_e32_teacher",
    "e36": "adapters/sql_lora_e36_teacher",
}

examples = json.loads(DATA.read_text())
assert len(examples) == 30

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def readonly(action, a, b, db, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return (
        sqlite3.SQLITE_OK
        if action in allowed
        else sqlite3.SQLITE_DENY
    )


def connect(db_id):
    path = DB_ROOT / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA trusted_schema = OFF")
    return conn


def get_schema(conn):
    return "\n".join(
        row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL"
        )
    )


def compact_schema(full_schema, question):
    # Exactly the same heuristic used during E27 training.
    tables = []
    for statement in full_schema.split("\n"):
        match = re.search(
            r'CREATE TABLE\s+["`]?(\w+)',
            statement,
            re.IGNORECASE,
        )
        if match and match.group(1).lower() in question.lower():
            tables.append(statement)

    return "\n".join(tables) if tables else full_schema


def execute(conn, sql):
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        raise ValueError("Expected a SELECT query")
    return conn.execute(sql).fetchall()


def equivalent(expected, actual, reference_sql):
    if re.search(r"\bORDER\s+BY\b", reference_sql, re.I):
        return expected == actual

    def norm(rows):
        return Counter(
            tuple(
                round(v, 6) if isinstance(v, float) else v
                for v in row
            )
            for row in rows
        )

    return norm(expected) == norm(actual)


def system_message(schema):
    return {
        "role": "system",
        "content": (
            "Translate the user's question into SQLite SQL. "
            "Return exactly one SELECT statement and no explanation. "
            "Return only the columns needed to answer the question.\n\n"
            f"Database schema:\n{schema}"
        ),
    }


def initial_messages(ex, schema):
    return [
        system_message(schema),
        {"role": "user", "content": ex["question"]},
    ]


def recovery_messages(ex, schema, failed_sql, error):
    return initial_messages(ex, schema) + [
        {"role": "assistant", "content": failed_sql},
        {
            "role": "user",
            "content": (
                f"SQLite error: {error}\n"
                "Fix the previous SQL using the schema. "
                "Preserve the question's requirements. "
                "Return only corrected SQL."
            ),
        },
    ]


def generate(model, messages):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


# Prepare reference results once; do not expose them to models.
tasks = []

for ex in examples:
    with connect(ex["db_id"]) as conn:
        schema = get_schema(conn) 
        conn.set_authorizer(readonly)
        expected = execute(conn, ex["reference_sql"])

    tasks.append({
        "db_id": ex["db_id"],
        "question": ex["question"],
        "reference_sql": ex["reference_sql"],
        "schema": schema,
        "expected": expected,
    })


def score(task, sql):
    with connect(task["db_id"]) as conn:
        conn.set_authorizer(readonly)

        try:
            actual = execute(conn, sql)
            passed = equivalent(
                task["expected"],
                actual,
                task["reference_sql"],
            )
            return passed, None

        except (sqlite3.Error, ValueError) as exc:
            return False, str(exc)


# First, all models get the same 30 questions.
# Then, all models receive the SAME failed v4 SQL and errors.
results = {}
shared_failures = []

for name, adapter in ADAPTERS.items():
    print(f"\n=== E28: {name.upper()} ===", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda",
    )

    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    first_attempts = []

    for i, task in enumerate(tasks, 1):
        sql = generate(
            model,
            initial_messages(task, task["schema"]),
        )

        passed, error = score(task, sql)

        first_attempts.append({
            "sql": sql,
            "passed": passed,
            "error": error,
        })

        print(
            f"{name} first [{i}/30]: "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    # Freeze the actual v4 execution-error cases.
    if name == "v4":
        shared_failures = [
            {
                "index": i,
                "failed_sql": r["sql"],
                "error": r["error"],
            }
            for i, r in enumerate(first_attempts)
            if r["error"] is not None
        ]

    recoveries = []

    for item in shared_failures:
        task = tasks[item["index"]]

        corrected_sql = generate(
            model,
            recovery_messages(
                task,
                task["schema"],
                item["failed_sql"],
                item["error"],
            ),
        )

        passed, error = score(task, corrected_sql)

        recoveries.append({
            "index": item["index"],
            "corrected_sql": corrected_sql,
            "passed": passed,
            "error": error,
        })

    results[name] = {
        "first_attempts": first_attempts,
        "recoveries": recoveries,
    }

    first_correct = sum(
        x["passed"] for x in first_attempts
    )
    recovered = sum(x["passed"] for x in recoveries)

    print(f"\n{name} first attempt: {first_correct}/30")
    print(
        f"{name} shared recovery: "
        f"{recovered}/{len(shared_failures)}"
    )

    del model, base
    gc.collect()
    torch.cuda.empty_cache()


# Save full traces for future teacher-data collection.
OUT.parent.mkdir(exist_ok=True)

OUT.write_text(
    json.dumps({
        "tasks": tasks,
        "shared_failures": shared_failures,
        "models": results,
    }, indent=2)
)

print("\n=== E28 FINAL COMPARISON ===")

for name, data in results.items():
    first = sum(
        x["passed"] for x in data["first_attempts"]
    )
    recovered = sum(
        x["passed"] for x in data["recoveries"]
    )

    print(
        f"{name:9} | First: {first}/30 "
        f"| Recovery: {recovered}/{len(shared_failures)}"
    )

print("Saved:", OUT)
