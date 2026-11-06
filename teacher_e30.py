
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
DB_ROOT = Path("data/spider_data/database")
from openai import OpenAI


# ============================================================
# E29 — Qwen3.5-9B Teacher: SQL Error Correction
# Student: Qwen3-0.6B (LoRA v4)
# Teacher: Qwen3.5-9B through Hugging Face
# ============================================================

MODEL = "Qwen/Qwen3.5-9B"
BASE_URL = "https://router.huggingface.co/v1"

INPUT = Path("data/distill/e46_teacher_input_2.json")
OUTPUT = Path("data/distill/e46_teacher_output_2.json")
MAX_CASES = 58


token = os.environ.get("HF_TOKEN")

if not token:
    raise RuntimeError(
        "HF_TOKEN is missing. Set it in your terminal."
    )

client = OpenAI(
    base_url=BASE_URL,
    api_key=token,
    timeout=120.0,
    max_retries=0,
)


# ============================================================
# Load student failures
# ============================================================

data = json.loads(INPUT.read_text())

tasks = data["tasks"]
failures = data["shared_failures"][:MAX_CASES]

print("\n=== E29 TEACHER EXPERIMENT ===")
print("Teacher:", MODEL)
print("Student:", "Qwen3-0.6B + LoRA v4")
print("Cases:", len(failures), flush=True)


# ============================================================
# SQLite helpers
# ============================================================

def readonly(action, arg1, arg2, dbname, source):
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


def get_schema(conn):
    return "\n".join(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL"
        )
    )


def normalize(rows):
    return Counter(
        tuple(
            round(value, 6)
            if isinstance(value, float)
            else value
            for value in row
        )
        for row in rows
    )


def clean_sql(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:sql)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```$", "", text).strip()

    return text


def execute(conn, sql):
    sql = sql.strip()

    if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Expected SELECT SQL")

    # sqlite3.execute rejects multiple statements.
    return conn.execute(sql).fetchall()


def equivalent(expected, actual, reference_sql):
    if re.search(
        r"\bORDER\s+BY\b",
        reference_sql,
        re.IGNORECASE,
    ):
        return expected == actual

    return normalize(expected) == normalize(actual)


# ============================================================
# Teacher prompt
# ============================================================

def make_prompt(question, schema, failed_sql, error):
    return (
        f"Database schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"Your student's previous SQL:\n{failed_sql}\n\n"
        f"SQLite execution error:\n{error}\n\n"
        "Task:\n"
        "Correct the SQL so that it answers the original question.\n"
        "Use only tables and columns present in the schema.\n"
        "Fix invalid joins, column references, and aliases.\n"
        "Preserve all required filters, aggregations, and ordering.\n"
        "Return exactly one SQLite SELECT query.\n"
        "No markdown, explanations, or commentary."
    )


# ============================================================
# Run teacher and independently verify responses
# ============================================================

records = []

for i, failure in enumerate(failures, 1):

    task = tasks[failure["index"]]

    db_id = task["db_id"]
    question = task["question"]

    db_path = (
        DB_ROOT
        / db_id
        / f"{db_id}.sqlite"
    )

    print(
        f"\n[{i}/{len(failures)}] Database: {db_id}",
        flush=True,
    )

    conn = None
    teacher_sql = ""
    verified = False
    verification_error = None
    expected = None
    actual = None

    try:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)

        conn = sqlite3.connect(
            f"file:{db_path.resolve()}?mode=ro",
            uri=True,
        )

        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA trusted_schema = OFF")

        # Retrieve full schema before applying authorizer.
        schema = get_schema(conn)

        conn.set_authorizer(readonly)

        prompt = make_prompt(
            question=question,
            schema=schema,
            failed_sql=failure["failed_sql"],
            error=failure["error"],
        )

        # Teacher receives no reference SQL or answer.
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert SQLite SQL assistant. "
                        "Correct SQL queries using the supplied "
                        "database schema and execution feedback."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=4096,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        print("Finish reason:", choice.finish_reason)
        print(
          "Output tokens:",
           response.usage.completion_tokens if response.usage else "unknown"
             )
        print("Raw response:", repr(content[:500]))

        teacher_sql = clean_sql(content)

        # Reference result used ONLY for verification.
        expected = execute(
            conn,
            task["reference_sql"],
        )

        actual = execute(
            conn,
            teacher_sql,
        )

        verified = equivalent(
            expected,
            actual,
            task["reference_sql"],
        )

        print("Question:", question)
        print("Student SQL:", failure["failed_sql"])
        print("Teacher SQL:", teacher_sql)
        print(
            "Verification:",
            "PASS" if verified else "FAIL",
            flush=True,
        )

    except Exception as exc:
        verification_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "Error:",
            verification_error,
            flush=True,
        )

    finally:
        if conn is not None:
            conn.close()

    record = {
        "db_id": db_id,
        "question": question,
        "failed_sql": failure["failed_sql"],
        "tool_feedback": failure["error"],
        "teacher_model": MODEL,
        "teacher_sql": teacher_sql,
        "verified": verified,
        "verification_error": verification_error,
    }

    records.append(record)

    # Save after each case to preserve completed results.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(records, indent=2)
    )


# ============================================================
# Final summary
# ============================================================

successful = sum(
    record["verified"]
    for record in records
)

api_errors = sum(
    record["verification_error"] is not None
    and not record["teacher_sql"]
    for record in records
)

print("\n=== E29 TEACHER SUMMARY ===")
print("Teacher:", MODEL)
print("Attempted:", len(records))
print("Verified:", successful)
print("API / generation failures:", api_errors)
print("Saved:", OUTPUT)
