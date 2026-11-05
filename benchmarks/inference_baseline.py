
import json
import statistics
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16
).to("cuda").eval()

messages = [{
    "role": "user",
    "content": (
        "A restaurant's revenue dropped 20% yesterday. "
        "Name three data checks to investigate the cause."
    )
}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to("cuda")

speeds = []
latencies = []

for run in range(4):
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output.shape[1] - inputs.shape[1]

    if run == 0:
        print("Warmup complete")
        continue

    speed = new_tokens / elapsed
    speeds.append(speed)
    latencies.append(elapsed)

    print(f"Run {run}: {speed:.2f} tok/s, {elapsed:.2f}s")

results = {
    "model": model_id,
    "runs": 3,
    "mean_tokens_per_second": round(statistics.mean(speeds), 2),
    "std_tokens_per_second": round(statistics.stdev(speeds), 2),
    "median_latency_seconds": round(statistics.median(latencies), 2),
    "max_allocated_mib": round(
        torch.cuda.max_memory_allocated() / 1024**2, 2
    ),
}

with open("results/experiment_0e.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
