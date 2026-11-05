
import json
import random
import sqlite3
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ROOT = Path("data/spider_data")
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT = Path("adapters/sql_lora_v3")
MAX_TOKENS = 512

random.seed(42)
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda",
)

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

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

model.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4,
)

schema_cache = {}

def prepare(path):
    examples = json.loads(Path(path).read_text())
    prepared = []

    for ex in examples:
        db_id = ex["db_id"]

        if db_id not in schema_cache:
            db = ROOT / "database" / db_id / f"{db_id}.sqlite"

            conn = sqlite3.connect(
                f"file:{db.resolve()}?mode=ro",
                uri=True,
            )
            schema_cache[db_id] = "\n".join(
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
            ex["sql"] + tokenizer.eos_token,
            add_special_tokens=False,
        )

        ids = prompt_ids + sql_ids
        labels = [-100] * len(prompt_ids) + sql_ids

        assert len(ids) <= MAX_TOKENS, (
            f"Sequence exceeds limit: {db_id}, {len(ids)} tokens"
        )

        prepared.append((ids, labels))

    return prepared


train_data = prepare("data/train_v3.json")
val_data = prepare("data/val_v3.json")

assert len(train_data) == 400
assert len(val_data) == 80

print("Training:", len(train_data), flush=True)
print("Validation:", len(val_data), flush=True)


def compute_loss(ids, labels):
    inputs = torch.tensor([ids], device="cuda")
    targets = torch.tensor([labels], device="cuda")

    output = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        labels=targets,
    )
    return output.loss


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for ids, labels in val_data:
        loss = compute_loss(ids, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")

        tokens = sum(x != -100 for x in labels)
        total_loss += loss.item() * tokens
        total_tokens += tokens

    model.train()
    return total_loss / total_tokens


best_val = float("inf")
train_losses = []

model.train()
torch.cuda.reset_peak_memory_stats()

for step, (ids, labels) in enumerate(train_data, 1):
    optimizer.zero_grad(set_to_none=True)

    loss = compute_loss(ids, labels)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: step {step}")

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if step % 20 == 0:
        print(
            f"Step {step}/400 | "
            f"Mean recent loss: "
            f"{sum(train_losses[-20:]) / 20:.4f}",
            flush=True,
        )

    if step in (200, 400):
        val_loss = validate()

        print(
            f"\nCHECKPOINT {step} | "
            f"Validation loss: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            OUTPUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT)
            tokenizer.save_pretrained(OUTPUT)

            print("New best adapter saved.", flush=True)

print("\n=== V3 TRAINING SUMMARY ===")
print("Training steps:", len(train_losses))
print("Mean training loss:", round(sum(train_losses) / len(train_losses), 4))
print("Best validation loss:", round(best_val, 4))
print(
    "Peak allocated VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    "MiB",
)
print("Best adapter:", OUTPUT)
