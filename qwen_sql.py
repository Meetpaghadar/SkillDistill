import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("PRAGMA query_only = ON")
# This read-only connection still needs a populated database.
# For now, initialize the data using a separate writable connection.

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()

import sqlite3

# Fixed evaluation dataset
conn = sqlite3.connect(":memory:")

conn.execute("""
CREATE TABLE orders (
    id INTEGER,
    restaurant TEXT,
    amount REAL,
    status TEXT
)
""")

conn.executemany(
    "INSERT INTO orders VALUES (?, ?, ?, ?)",
    [
        (1, "A", 100, "completed"),
        (2, "A", 200, "completed"),
        (3, "A", 50, "cancelled"),
        (4, "B", 150, "completed"),
        (5, "B", 80, "cancelled"),
    ],
)

conn.execute("PRAGMA query_only = ON")


def readonly_authorizer(action, arg1, arg2, dbname, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


conn.set_authorizer(readonly_authorizer)

tests = [
    ("What is the total revenue from all completed orders?", [(450.0,)]),
    ("How many orders does restaurant A have, regardless of status?", [(3,)]),
    ("What is the average amount of cancelled orders?", [(65.0,)]),
    ("What is the largest completed order amount?", [(200.0,)]),
    ("How many distinct restaurants have completed orders?", [(2,)]),
]

passed_count = 0

for i, (question, expected) in enumerate(tests, start=1):
    messages = [
        {
            "role": "system",
            "content": (
               "Translate the user's question into SQLite SQL. "
"Return only one SELECT statement. No explanation.\n"
"Table: orders(id INTEGER, restaurant TEXT, "
"amount REAL, status TEXT).\n"
"Rules:\n"
"1. Include every filtering condition mentioned in the question.\n"
"2. Completed orders require status = 'completed'.\n"
"3. Cancelled orders require status = 'cancelled'.\n"
"4. Total revenue requires SUM(amount).\n"
"5. Average order amount requires AVG(amount).\n"
"6. Count questions require COUNT(*)."
"7. Return exactly the columns needed to answer the question. "
"Do not include additional metrics or columns."
            ),
        },
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    sql = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    try:
        actual = conn.execute(sql).fetchall()
        passed = actual == expected
        error = None
    except sqlite3.Error as exc:
        actual = None
        passed = False
        error = str(exc)

    passed_count += int(passed)

    print(f"\n--- TEST {i} ---")
    print("Question:", question)
    print("Generated SQL:", sql)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Result:", "PASS" if passed else "FAIL")

    if error:
        print("SQL error:", error)

print("\n=== BASELINE SUMMARY ===")
print(f"Passed: {passed_count}/{len(tests)}")
print(f"Success rate: {passed_count / len(tests):.0%}")

conn.close()