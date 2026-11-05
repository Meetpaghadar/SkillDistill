
import gc
import json
import random
import sqlite3
from pathlib import Path
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# E27: Control vs Recovery Fine-Tuning
# Both experiments start from the same LoRA v4 adapter.
# ============================================================

MODEL_ID = "Qwen/Qwen3-0.6B"
START_ADAPTER = "adapters/sql_lora_v4"

DATA_ROOT = Path("data/distill")
DB_ROOT = Path("data/spider_data/database")

MAX_TOKENS = 1537
EPOCHS = 2
LEARNING_RATE = 5e-5
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "CUDA GPU required"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

control = json.loads(
    (DATA_ROOT / "e27_control.json").read_text()
)

recovery = json.loads(
    (DATA_ROOT / "e27_recovery.json").read_text()
)

assert len(control) == len(recovery) == 30


# ============================================================
# Schema and conversation formatting
# ============================================================

schema_cache = {}


def get_schema(db_id):
    if db_id in schema_cache:
        return schema_cache[db_id]

    path = DB_ROOT / db_id / f"{db_id}.sqlite"

    if not path.is_file():
        raise FileNotFoundError(path)

    with sqlite3.connect(
        f"file:{path.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        schema = "\n".join(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL"
            )
        )

    schema_cache[db_id] = schema
    return schema

def compact_schema(db_id, question):
    full_schema = get_schema(db_id)

    # Keep only CREATE TABLE definitions whose table names
    # appear in the question. Fall back to full schema.
    tables = []
    for statement in full_schema.split("\n"):
        match = re.search(
            r'CREATE TABLE\s+["`]?(\w+)',
            statement,
            re.IGNORECASE,
        )
        if match and match.group(1).lower() in question.lower():
            tables.append(statement)

    return "\n".join(tables) if tables else full_schema


def make_messages(example, mode):
    schema = get_schema(example["db_id"])


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
        {
            "role": "user",
            "content": example["question"],
        },
    ]

    if mode == "recovery":
        messages.extend([
            {
                "role": "assistant",
                "content": example["failed_sql"],
            },
         {
    "role": "user",
    "content": (
        f"SQLite error: {example['tool_feedback']}\n"
        "Fix the previous SQL using the schema. "
        "Preserve the question's requirements. "
        "Return only corrected SQL."
    ),
}
        ])

    return messages


def encode(example, mode):
    prompt_ids = tokenizer.apply_chat_template(
        make_messages(example, mode),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    target_ids = tokenizer.encode(
        example["target_sql"] + tokenizer.eos_token,
        add_special_tokens=False,
    )

    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(ids) > MAX_TOKENS:
        return None

    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


# ============================================================
# Match examples before training.
# Both treatments must use exactly the same questions.
# ============================================================

print("\n=== TOKEN LENGTH DIAGNOSTIC ===")

for mode, data in [("control", control), ("recovery", recovery)]:
    lengths = []

    for ex in data:
        prompt = tokenizer.apply_chat_template(
            make_messages(ex, mode),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = tokenizer.encode(
            ex["target_sql"] + tokenizer.eos_token,
            add_special_tokens=False,
        )
        lengths.append(len(prompt) + len(target))

    lengths.sort()

    print(
        f"{mode}: min={lengths[0]}, "
        f"median={lengths[len(lengths)//2]}, "
        f"max={lengths[-1]}, "
        f"<=512={sum(n <= 512 for n in lengths)}/30, "
        f"<=768={sum(n <= 768 for n in lengths)}/30"
    )

paired = []

for a, b in zip(control, recovery):
    assert a["db_id"] == b["db_id"]
    assert a["question"] == b["question"]
    assert a["target_sql"] == b["target_sql"]

    encoded_control = encode(a, "control")
    encoded_recovery = encode(b, "recovery")

    if encoded_control is None or encoded_recovery is None:
        continue

    paired.append((encoded_control, encoded_recovery))

print("\n=== E27 DATA CHECK ===", flush=True)
print("Original pairs:", len(control))
print("Pairs fitting both prompts:", len(paired))
print("Pairs excluded:", len(control) - len(paired))
print("Starting GPU memory test...")

if len(paired) < 10:
    raise RuntimeError(
        "Too few matched examples fit within 512 tokens. "
        "Stop and inspect token lengths before training."
    )


# ============================================================
# Train one independent experiment
# ============================================================

def train(mode, pair_index, output_path):
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"\n=== TRAINING {mode.upper()} ===", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda",
    )

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        START_ADAPTER,
        is_trainable=True,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
    )

    data = [pair[pair_index] for pair in paired]

    losses = []
    step = 0

    for epoch in range(EPOCHS):
        order = list(range(len(data)))
        random.shuffle(order)

        for index in order:
            ids, labels = data[index]

            ids = ids.to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad(set_to_none=True)

            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                labels=labels,
                use_cache=False,
            )

            loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                1.0,
            )

            optimizer.step()
            if step == 1:
              print(
                     f"Peak GPU memory: "
                     f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB"
    )
            raise SystemExit("Memory test complete.")

            step += 1
            losses.append(loss.item())

            if step % 10 == 0:
                print(
                    f"{mode} | Step {step}/{len(data) * EPOCHS} "
                    f"| Loss {loss.item():.4f}",
                    flush=True,
                )

    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"\n{mode.upper()} COMPLETE")
    print("Steps:", step)
    print("Mean loss:", round(sum(losses) / len(losses), 4))
    print("Saved:", output_path)

    del optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Run both experiments independently
# ============================================================

train(
    mode="control",
    pair_index=0,
    output_path=Path("adapters/sql_lora_e27_control"),
)

train(
    mode="recovery",
    pair_index=1,
    output_path=Path("adapters/sql_lora_e27_recovery"),
)

print("\n=== E27 FINISHED ===")
print("Control: adapters/sql_lora_e27_control")
print("Recovery: adapters/sql_lora_e27_recovery")
print("Original v4 adapter preserved.")
