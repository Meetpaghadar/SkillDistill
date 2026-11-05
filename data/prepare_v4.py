
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("data/spider_data")
SEED = 42
MAX_TOKENS = 512
TRAIN_SIZE = 1000
VAL_SIZE = 80

rng = random.Random(SEED)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

examples = json.loads((ROOT / "train_spider.json").read_text())

SKILLS = {
    "JOIN": r"\bJOIN\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "DISTINCT": r"\bDISTINCT\b",
    "COUNT": r"\bCOUNT\s*\(",
    "SUM": r"\bSUM\s*\(",
    "AVG": r"\bAVG\s*\(",
    "MIN": r"\bMIN\s*\(",
    "MAX": r"\bMAX\s*\(",
    "SUBQUERY": r"\(\s*SELECT\b",
}

# Match the prompt used for inference and training.
def make_prompt(ex, schema):
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

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# First: filter for complete, executable examples.
eligible = []
seen = set()
schemas = {}
connections = {}

try:
    for ex in examples:
        db_id = ex["db_id"]
        key = (db_id, " ".join(ex["question"].lower().split()))

        if key in seen:
            continue

        db_path = ROOT / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            continue

        if db_id not in connections:
            conn = sqlite3.connect(
                f"file:{db_path.resolve()}?mode=ro",
                uri=True,
            )
            conn.execute("PRAGMA query_only = ON")
            connections[db_id] = conn

            schemas[db_id] = "\n".join(
                row[0]
                for row in conn.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='table' AND sql IS NOT NULL"
                )
            )

        prompt_ids = make_prompt(ex, schemas[db_id])
        sql_ids = tokenizer.encode(
            ex["query"] + tokenizer.eos_token,
            add_special_tokens=False,
        )

        if len(prompt_ids) + len(sql_ids) > MAX_TOKENS:
            continue

        try:
            connections[db_id].execute(ex["query"]).fetchmany(1)
        except sqlite3.Error:
            continue

        seen.add(key)
        eligible.append({
            "db_id": db_id,
            "question": ex["question"],
            "sql": ex["query"],
        })
finally:
    for conn in connections.values():
        conn.close()

print("Eligible examples:", len(eligible))

# Second: split by database, not by individual question.
groups = defaultdict(list)
for ex in eligible:
    groups[ex["db_id"]].append(ex)

db_ids = sorted(groups)
rng.shuffle(db_ids)

eligible_val_dbs = [
    db for db in db_ids
    if len(groups[db]) >= 8
]

assert len(eligible_val_dbs) >= 10, (
    "Not enough databases with 8 eligible examples"
)

# Reuse exactly the validation databases from v3.
frozen_validation = json.loads(
    Path("data/val_v3.json").read_text()
)

val_dbs = {ex["db_id"] for ex in frozen_validation}

assert len(val_dbs) == 10

train_pool = [
    ex for ex in eligible if ex["db_id"] not in val_dbs
]
val_pool = [
    ex for ex in eligible if ex["db_id"] in val_dbs
]

assert len(train_pool) >= TRAIN_SIZE
assert len(val_pool) >= VAL_SIZE

# Third: reserve validation examples.
# Sample validation examples across reserved schemas.
# Preserve the identical 80 validation questions.
validation = frozen_validation

assert len(validation) == VAL_SIZE
assert all(ex["db_id"] in val_dbs for ex in validation)
# Fourth: sample training examples with skill coverage.
rng.shuffle(train_pool)

selected = []
used = set()
db_counts = Counter()

# Target up to 20 examples per SQL construct.
for skill, pattern in SKILLS.items():
    candidates = [
        ex for ex in train_pool
        if re.search(pattern, ex["sql"], re.I)
        and (ex["db_id"], ex["question"]) not in used
    ]

    # Prefer databases with fewer selected examples.
    candidates.sort(key=lambda ex: db_counts[ex["db_id"]])

    for ex in candidates[:20]:
        key = (ex["db_id"], ex["question"])
        selected.append(ex)
        used.add(key)
        db_counts[ex["db_id"]] += 1

# Fill remaining slots with a diverse random sample.
remaining = [
    ex for ex in train_pool
    if (ex["db_id"], ex["question"]) not in used
]

rng.shuffle(remaining)
remaining.sort(key=lambda ex: db_counts[ex["db_id"]])

for ex in remaining:
    if len(selected) >= TRAIN_SIZE:
        break
    selected.append(ex)
    db_counts[ex["db_id"]] += 1

training = selected[:TRAIN_SIZE]

assert len(training) == TRAIN_SIZE
assert len(validation) == VAL_SIZE

train_dbs = {ex["db_id"] for ex in training}
validation_dbs = {ex["db_id"] for ex in validation}

assert train_dbs.isdisjoint(validation_dbs)

# Save reproducible datasets.
Path("data/train_v4.json").write_text(
    json.dumps(training, indent=2)
)
Path("data/val_v4.json").write_text(
    json.dumps(validation, indent=2)
)

print("\n=== V4 DATASET SUMMARY ===")
print("Training examples:", len(training))
print("Validation examples:", len(validation))
print("Training databases:", len(train_dbs))
print("Validation databases:", len(validation_dbs))
print("Overlapping databases:", len(train_dbs & validation_dbs))

print("\nTraining SQL skill coverage:")
for skill, pattern in SKILLS.items():
    count = sum(
        bool(re.search(pattern, ex["sql"], re.I))
        for ex in training
    )
    print(f"{skill:<12} {count}/{len(training)}")

print("\nSaved: data/train_v3.json")
print("Saved: data/val_v3.json")
