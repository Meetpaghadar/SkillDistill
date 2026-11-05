import json
import sqlite3
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

root = Path("data/spider_data")
examples = json.loads((root / "train_spider.json").read_text())
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

schema_cache = {}
counts = Counter()

for ex in examples:
    db_id = ex["db_id"]

    if db_id not in schema_cache:
        path = root / "database" / db_id / f"{db_id}.sqlite"

        conn = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
        )

        schema_cache[db_id] = "\n".join(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL"
            )
        )
        conn.close()

    messages = [
        {
            "role": "system",
            "content": (
                "Translate the user's question into SQLite SQL. "
                "Return exactly one SELECT statement and no explanation. "
                "Return only the columns needed to answer the question.\n\n"
                f"Database schema:\n{schema_cache[db_id]}"
            ),
        },
        {"role": "user", "content": ex["question"]},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    sql_ids = tokenizer.encode(
        ex["query"] + tokenizer.eos_token,
        add_special_tokens=False,
    )

    length = len(prompt_ids) + len(sql_ids)

    counts["total"] += 1

    if length <= 512:
        counts["fits_512"] += 1
    elif length <= 768:
        counts["fits_768"] += 1
    elif length <= 1024:
        counts["fits_1024"] += 1
    else:
        counts["over_1024"] += 1

print("\n=== TRAINING CAPACITY ===")
for key, value in counts.items():
    print(f"{key}: {value}")

print(
    "Percentage fitting 512 tokens:",
    round(100 * counts["fits_512"] / counts["total"], 1),
    "%",
)