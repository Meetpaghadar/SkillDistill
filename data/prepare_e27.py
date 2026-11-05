
import json
from pathlib import Path

root = Path("data/distill")
examples = json.loads((root / "recovery_train.json").read_text())

control = []
recovery = []

for x in examples:
    target = x["corrected_sql"]

    control.append({
        "db_id": x["db_id"],
        "question": x["question"],
        "target_sql": target,
    })

    recovery.append({
        "db_id": x["db_id"],
        "question": x["question"],
        "failed_sql": x["failed_sql"],
        "tool_feedback": x["tool_feedback"],
        "target_sql": target,
    })

assert len(control) == len(recovery) == 30

(root / "e27_control.json").write_text(
    json.dumps(control, indent=2)
)

(root / "e27_recovery.json").write_text(
    json.dumps(recovery, indent=2)
)

print("=== E27 DATA READY ===")
print("Control:", len(control))
print("Recovery:", len(recovery))
print("Shared targets:", all(
    a["target_sql"] == b["target_sql"]
    for a, b in zip(control, recovery)
))
