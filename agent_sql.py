
import sqlite3
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
DB = Path("data/spider_data/database/flight_2/flight_2.sqlite")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda",
)

model = PeftModel.from_pretrained(
    base,
    "adapters/sql_lora_v4",
)
model.eval()

conn = sqlite3.connect(f"file:{DB.resolve()}?mode=ro", uri=True)
conn.execute("PRAGMA query_only = ON")
conn.execute("PRAGMA trusted_schema = OFF")

schema = "\n".join(
    row[0]
    for row in conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL"
    )
)

def readonly(action, arg1, arg2, dbname, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

conn.set_authorizer(readonly)


def generate(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


question = 'Which country does Airline "JetBlue Airways" belong to?'

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
    {"role": "user", "content": question},
]

# First attempt
sql = generate(messages)
print("\nATTEMPT 1:", sql, flush=True)

def execute(sql):
    if not sql.lstrip().upper().startswith("SELECT "):
        raise ValueError("Only SELECT queries are allowed")
    return conn.execute(sql).fetchall()

try:
    result = execute(sql)
    print("TOOL RESULT:", result, flush=True)

except (sqlite3.Error, ValueError) as error:
    print("TOOL ERROR:", error, flush=True)

    # Give the model its previous SQL and the tool error.
    messages.append({"role": "assistant", "content": sql})
    messages.append({
        "role": "user",
        "content": (
            f"SQLite returned this error: {error}\n"
            "Correct your SQL using the database schema. "
            "Return only the corrected SQL statement."
        ),
    })

    corrected_sql = generate(messages)
    print("\nATTEMPT 2:", corrected_sql, flush=True)

    try:
        result = execute(corrected_sql)
        print("TOOL RESULT:", result, flush=True)
    except (sqlite3.Error, ValueError) as second_error:
        print("TOOL ERROR:", second_error, flush=True)

conn.close()
