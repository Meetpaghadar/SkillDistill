
import json
import random
from collections import defaultdict
from pathlib import Path

rng = random.Random(42)
out = Path("data/distill")
out.mkdir(parents=True, exist_ok=True)

def load(path):
    return json.loads(Path(path).read_text())

# Exclude all databases already used in v4 training or validation.
seen = {
    x["db_id"]
    for path in ["data/train_v4.json", "data/val_v4.json"]
    for x in load(path)
}

spider = load("data/spider_data/train_spider.json")
by_db = defaultdict(list)

for x in spider:
    if x["db_id"] not in seen and x.get("question") and x.get("query"):
        by_db[x["db_id"]].append({
            "db_id": x["db_id"],
            "question": x["question"],
            "reference_sql": x["query"],
        })

eligible = [db for db, rows in by_db.items() if len(rows) >= 10]
rng.shuffle(eligible)

if len(eligible) < 16:
    raise RuntimeError(
        f"Only {len(eligible)} eligible unseen databases. "
        "Need a different source or smaller split."
    )

test_dbs = eligible[:10]
val_dbs = eligible[10:15]
train_dbs = eligible[15:]

def sample(dbs, count):
    pool = [x for db in dbs for x in by_db[db]]
    rng.shuffle(pool)
    if len(pool) < count:
        raise RuntimeError(f"Need {count} examples, found {len(pool)}")
    return pool[:count]

splits = {
    "train": sample(train_dbs, 100),
    "val": sample(val_dbs, 30),
    "test": sample(test_dbs, 50),
}

for name, data in splits.items():
    (out / f"{name}.json").write_text(json.dumps(data, indent=2))
    dbs = {x["db_id"] for x in data}
    assert dbs.isdisjoint(seen)
    print(f"{name}: {len(data)} questions | {len(dbs)} databases")

sets = [
    {x["db_id"] for x in data}
    for data in splits.values()
]
assert not (sets[0] & sets[1])
assert not (sets[0] & sets[2])
assert not (sets[1] & sets[2])

print("Previous v4 database overlap: 0")
print("Cross-split schema overlap: 0")
print("E24 clean split: PASSED")
