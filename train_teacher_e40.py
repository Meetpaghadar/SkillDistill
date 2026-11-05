
import json
import random
import sqlite3
from pathlib import Path
DB_ROOT = Path("data/spider_data/database")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
START_ADAPTER = "adapters/sql_lora_v4"
OUTPUT = Path("adapters/sql_lora_e32_teacher")
DATA = Path("data/distill/e36_teacher_train.json")
OUTPUT = Path("adapters/sql_lora_e40_overfit")

MAX_TOKENS = 1536
EPOCHS = 10
LR = 5e-5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
examples = json.loads(DATA.read_text())
examples = examples[:5]

assert len(examples) >=15


def get_schema(db_id):
    path = DB_ROOT / db_id / f"{db_id}.sqlite"

    with sqlite3.connect(
        f"file:{path.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        return "\n".join(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL"
            )
        )


def prepare(example):
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
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    target = tokenizer.encode(
        example["target_sql"] + tokenizer.eos_token,
        add_special_tokens=False,
    )

    ids = prompt + target
    labels = [-100] * len(prompt) + target

    if len(ids) > MAX_TOKENS:
        raise ValueError(
            f"{example['db_id']}: {len(ids)} tokens "
            f"exceeds {MAX_TOKENS}"
        )

    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


train_data = [prepare(ex) for ex in examples]

print("\n=== E32 DATA CHECK ===")
print("Teacher examples:", len(train_data))
print("Longest:", max(x[0].shape[1] for x in train_data))
print("Shortest:", min(x[0].shape[1] for x in train_data))

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
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.train()

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
)

# Start with the longest example to detect memory problems early.
order = sorted(
    range(len(train_data)),
    key=lambda i: train_data[i][0].shape[1],
    reverse=True,
)

losses = []
step = 0

for epoch in range(EPOCHS):
    if epoch:
        random.shuffle(order)

    for index in order:
        ids, labels = train_data[index]
        ids = ids.cuda()
        labels = labels.cuda()

        optimizer.zero_grad(set_to_none=True)

        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            use_cache=False,
        )

        loss = output.loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            1.0,
        )
        optimizer.step()

        step += 1
        losses.append(loss.item())

        print(
            f"Step {step}/{len(train_data) * EPOCHS} "
            f"| Loss {loss.item():.4f}",
            flush=True,
        )

OUTPUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

print("\n=== E32 DISTILLATION SUMMARY ===")
print("Teacher:", "Qwen3.5-9B")
print("Student:", "Qwen3-0.6B")
print("Training examples:", len(train_data))
print("Steps:", step)
print("Mean loss:", round(sum(losses) / len(losses), 4))
print(
    "Peak VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**2),
    "MiB",
)
print("Adapter:", OUTPUT)
