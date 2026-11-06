# SkillDistill

**Teaching a small language model to recover from SQL errors using teacher–student distillation and execution feedback.**

SkillDistill explores whether a small, specialized LLM can learn useful SQL generation and error-recovery behavior from a stronger teacher, reducing the need to invoke larger models for every routine task.

Rather than training only on correct SQL answers, the project uses a **failure-driven learning pipeline**: the student attempts a query, SQLite returns execution feedback, and a larger teacher proposes a correction that is verified before being used for training.


## Approach

### Stage 1 — LoRA Fine-Tuning

Started with **Qwen3-0.6B** and performed parameter-efficient fine-tuning using LoRA/PEFT to establish a specialized SQL baseline.

### Stage 2 — Teacher–Student Distillation

Extended the baseline using:

- **Student:** Qwen3-0.6B + existing LoRA V4 adapter
- **Teacher:** Qwen3.5-9B
- **Dataset:** Spider text-to-SQL
- **Feedback:** SQLite execution errors
- **Training:** Supervised LoRA fine-tuning

### Failure-Driven Learning Pipeline

```text
Natural language question + Database schema
                    |
                    v
             Student generates SQL
                    |
                    v
                Execute SQL
                    |
                    v
          Capture execution failure
                    |
                    v
       Teacher receives question,
       schema, failed SQL and error
                    |
                    v
           Teacher corrects SQL
                    |
                    v
        Execute + verify correction
                    |
                    v
       Build verified training examples
                    |
                    v
          Fine-tune student (LoRA)
                    |
                    v
       Evaluate on held-out databases
```

Each training example includes the original question, failed student SQL, database error message, and verified teacher-generated correction.

The teacher is not provided the reference answer when generating a correction. The reference SQL is used afterward for execution-based verification.

## Experiments and Results

We compared independently trained student checkpoints using held-out database schemas.

### First-Attempt SQL Accuracy

| Model | Method | Test accuracy |
|---|---|---:|
| Original Qwen3-0.6B | No task-specific fine-tuning | 18% |
| V4 | Initial LoRA fine-tuning | 27% | |
| **E46** | **Distillation with 113 verified corrections** | **44%** |

### SQL Error Recovery

All checkpoints received the same failed SQL queries and SQLite error messages produced by V4.

| Model | Recovery accuracy |
|---|---:|
| V4 | 4.3% | |
| **E46** | **21.7%** |
 |
## Key Engineering Findings

**1. Feedback-based supervision**

The student was trained using failed queries, execution feedback, and teacher corrections, rather than correct SQL answers alone.

**2. Data quality matters**

Teacher-generated corrections were checked against reference execution results before entering the training dataset. More examples did not guarantee improved performance on every benchmark.

**3. Training loss isn't enough**

The larger dataset achieved lower training loss, but evaluation results varied across development and test sets. Model selection was based on measured outcomes rather than training loss alone.


## Implementation

Key files:

| File | Purpose |
|---|---|
| `collect_failures.py` | Collect failed SQL generations and execution errors |
| `teacher_e30.py` | Generate teacher corrections and verify their results |
| `train_teacher_e42.py` | Fine-tune using 50 verified corrections |
| `train_teacher_e46.py` | Fine-tune using 113 verified corrections |
| `evaluate_e47.py` | Compare student checkpoints on the held-out test set |
| `data/distill/e46_train_113.json` | Final training dataset |
| `results/e47_comparison.json` | Final benchmark results |

The training scripts use the Hugging Face Transformers and PEFT libraries with PyTorch. Teacher inference uses the Hugging Face inference router.

The code currently uses experiment-specific file paths and local LoRA checkpoints. To reproduce training or evaluation, the required Spider database files and starting adapters must first be prepared locally.

## Limitations

- Evaluation uses small development and test benchmarks.
- Execution-result agreement is not equivalent to complete SQL semantic correctness.
- The dataset contains teacher-generated supervision verified on individual database instances.
- Dataset size and total training steps increased together, so improvements cannot be attributed solely to additional examples.
- Inference latency, production cost, and real-world deployment reliability have not yet been benchmarked.

## Future Directions

- Evaluate stronger teachers such as DeepSeek, Kimi, and frontier API models.
- Improve supervision quality and error diversity.
- Strengthen SQL verification beyond single-database execution matching.
- Explore a local-first SQL agent that handles routine requests with a specialized student and escalates difficult cases to stronger models.
- Benchmark latency, inference cost, and recovery reliability.


**How much useful capability can we transfer from a large model into a small, specialized model—and how do we know the transfer actually worked?**

SkillDistill is an experimental exploration of that question through fine-tuning, teacher–student distillation, execution feedback, and controlled evaluation.