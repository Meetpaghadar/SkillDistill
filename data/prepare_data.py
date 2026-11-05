import json
import sqlite3
from pathlib import Path

root = Path("data/spider_data")
examples = json.loads((root / "train_spider.json").read_text())

verified = []
seen = set()

for ex in examples:
    db = ex["db_id"]
    question = ex["question"]
    sql = ex["query"]

    if (db, question) in seen:
        continue

    path = root / "database" / db / f"{db}.sqlite"
    if not path.exists():
        continue

    try:
        conn = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
        )
        conn.execute("PRAGMA query_only = ON")

        # Execute the reference SQL as a basic validity check.
        conn.execute(sql).fetchall()
        conn.close()

    except sqlite3.Error:
        continue

    seen.add((db, question))
    verified.append({
        "db_id": db,
        "question": question,
        "sql": sql,
    })

    if len(verified) == 20:
        break

Path("data/train_candidates.json").write_text(
    json.dumps(verified, indent=2)
)

print("Executable reference examples:", len(verified))
print("Databases represented:", len(set(x["db_id"] for x in verified)))

for ex in verified[:3]:
    print("\nQuestion:", ex["question"])
    print("SQL:", ex["sql"])