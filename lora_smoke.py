
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="cuda",
)

model.config.use_cache = False

# Freeze original weights, train only small LoRA adapters.
config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model.train()

# One tiny synthetic example for memory testing only.
messages = [
    {"role": "user", "content": "Translate to SQL: Count all singers."},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM singer;"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_tensors="pt",
)

# Training requires labels so cross-entropy loss can be computed.
inputs = {k: v.to("cuda") for k, v in inputs.items()}
inputs["labels"] = inputs["input_ids"].clone()

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4,
)

torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(set_to_none=True)

outputs = model(**inputs)
loss = outputs.loss
print("Initial loss:", round(loss.item(), 4), flush=True)

loss.backward()
optimizer.step()

print("Optimizer step: PASSED")
print(
    "Peak allocated VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    "MiB",
)
print(
    "Peak reserved VRAM:",
    round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    "MiB",
)
