
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path("data/spider_data")
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

# Validate reference SQL and remove exact duplicate pairs.
valid = []
seen = set()

for ex in examples:
    key = (
        ex["db_id"],
        " ".join(ex["question"].lower().split()),
    )
    if key in seen:
        continue

    db = ex["db_id"]
    path = ROOT / "database" / db / f"{db}.sqlite"

    if not path.is_file():
        continue

    conn = None
    try:
        conn = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
        )
        conn.execute("PRAGMA query_only = ON")
        conn.execute(ex["query"]).fetchmany(1)
    except sqlite3.Error:
        continue
    finally:
        if conn is not None:
            conn.close()

    seen.add(key)
    valid.append(ex)

# Greedy coverage: favor underrepresented SQL operations
# while also encouraging database diversity.
selected = []
used = set()
skill_counts = Counter()
db_counts = Counter()

for _ in range(min(100, len(valid))):
    best_index = None
    best_score = float("-inf")

    for index, ex in enumerate(valid):
        if index in used:
            continue

        present = [
            skill
            for skill, pattern in SKILLS.items()
            if re.search(pattern, ex["query"], re.I)
        ]

        skill_score = sum(
            1 / (1 + skill_counts[skill])
            for skill in present
        )
        db_score = 2 / (1 + db_counts[ex["db_id"]])
        score = skill_score + db_score

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        break

    ex = valid[best_index]
    selected.append(ex)
    used.add(best_index)

    db_counts[ex["db_id"]] += 1
    for skill, pattern in SKILLS.items():
        if re.search(pattern, ex["query"], re.I):
            skill_counts[skill] += 1

output = Path("data/train_candidates_v2.json")
output.write_text(json.dumps([
    {
        "db_id": ex["db_id"],
        "question": ex["question"],
        "sql": ex["query"],
    }
    for ex in selected
], indent=2))

print("Selected examples:", len(selected))
print("Unique databases:", len(db_counts))

print("\nSkill coverage:")
for skill in SKILLS:
    print(f"{skill:<12} {skill_counts[skill]}/{len(selected)}")

print("\nSaved:", output)
