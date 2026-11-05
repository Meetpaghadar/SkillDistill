
import json
import random
from pathlib import Path

rng = random.Random(42)

ROOT = Path("data/distill")
failures = json.loads(
    (ROOT / "student_failures.json").read_text()
)

# Prioritize execution errors, but include semantic failures.
execution = [x for x in failures if x["student_error"]]
semantic = [x for x in failures if not x["student_error"]]

rng.shuffle(execution)
rng.shuffle(semantic)

# Small pilot: 20 execution failures + 10 semantic failures.
selected = execution[:20] + semantic[:10]
rng.shuffle(selected)

assert len(selected) == 30

# The reference query is the supervision target.
# Do NOT include reference_sql in the student input.
records = []

for x in selected:
    records.append({
        "db_id": x["db_id"],
        "question": x["question"],
        "failed_sql": x["generated_sql"],
        "tool_feedback": (
            x["student_error"]
            if x["student_error"]
            else "Query executed, but its result was incorrect."
        ),
        "corrected_sql": x["reference_sql"],
        "failure_type": (
            "execution"
            if x["student_error"]
            else "semantic"
        ),
    })

path = ROOT / "recovery_train.json"
path.write_text(json.dumps(records, indent=2))

print("=== E26 SUMMARY ===")
print("Recovery examples:", len(records))
print("Execution errors:", sum(
    x["failure_type"] == "execution" for x in records
))
print("Semantic errors:", sum(
    x["failure_type"] == "semantic" for x in records
))
print("Saved:", path)
