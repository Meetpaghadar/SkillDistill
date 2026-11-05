
import json
import sqlite3
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ROOT = Path("data/spider_data")
examples = json.loads(
    Path("data/train_candidates.json").read_text()
)

MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda",
)
model.config.use_cache = False

model = get_peft_model(
    model,
    LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    ),
)

model.train()
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4,
)

# Prepare schema-conditioned training examples.
training_data = []

for ex in examples:
    db = ROOT / "database" / ex["db_id"] / (
        ex["db_id"] + ".sqlite"
    )

    conn = sqlite3.connect(
        f"file:{db.resolve()}?mode=ro",
        uri=True,
    )

    schema = "\n".join(
        row[0] for row in conn.execute(
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
                f"Database schema:\n{schema}"
            ),
        },
        {"role": "user", "content": ex["question"]},
    ]

    # Match the format used during evaluation.
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Only the SQL completion contributes to loss.
    sql_ids = tokenizer.encode(
        ex["sql"] + tokenizer.eos_token,
        add_special_tokens=False,
    )

    input_ids = prompt_ids + sql_ids
    labels = [-100] * len(prompt_ids) + sql_ids

    training_data.append((input_ids, labels))

print("Training examples:", len(training_data), flush=True)

torch.manual_seed(42)
torch.cuda.reset_peak_memory_stats()

losses = []

# One epoch, batch size 1.
for step, (ids, labels) in enumerate(training_data, 1):
    inputs = torch.tensor([ids], device="cuda")
    targets = torch.tensor([labels], device="cuda")

    optimizer.zero_grad(set_to_none=True)

    output = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        labels=targets,
    )

    loss = output.loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at step {step}")

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(
        f"Step {step}/{len(training_data)} "
        f"| Loss: {loss.item():.4f}",
        flush=True,
    )

print("\n=== TRAINING SUMMARY ===")
print("First loss:", round(losses[0], 4))
print("Last loss:", round(losses[-1], 4))
print("Mean loss:", round(sum(losses) / len(losses), 4))
print(
    "Peak allocated VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    "MiB",
)

model.save_pretrained("adapters/sql_lora_v1")
tokenizer.save_pretrained("adapters/sql_lora_v1")

print("Adapter saved: adapters/sql_lora_v1")
