# Off-policy distillation on MATH

This project uses two separate stages for top-K soft-target cross-entropy on
fixed MATH solutions:

1. `precompute_teacher_targets.py` runs the teacher and saves its top-K token
   log probabilities.
2. `train_off_policy.py` loads the saved targets and trains only the student.

The student is `Qwen/Qwen3.5-0.8B-Base`. The off-policy teacher is
`Qwen/Qwen3.5-4B`. There are no student rollouts.

## Experiment

- All 7,500 unique training examples from the seven
  `EleutherAI/hendrycks_math` subjects
- 10,000 total training prompts; the data loader repeats 2,500 examples after
  the first complete pass
- 2,048-token training limit
- Effective batch size 25
- 400 optimizer steps
- Evaluation after every 1,000 training prompts, at steps 40 through 400
- 50 held-out examples from `HuggingFaceH4/MATH-500`
- Exactly 10 evaluation examples from each difficulty level
- Seven or eight evaluation examples from each subject
- One or two examples from every subject-level group
- 1,024 generated tokens per evaluation example
- LoRA rank 32
- Teacher top-20 probabilities for off-policy distillation

The teacher uses bfloat16 weights without quantization during target
precomputation. This gives more exact soft targets. Set
`teacher.quantization: int8` and `teacher.dtype: float16` for a lower-memory test.
The student training stage does not load the teacher.

## Loss

For each fixed solution token, the teacher produces its 20 highest-probability
tokens. Their probabilities are normalized over the selected set. The student
minimizes cross-entropy against these soft targets. Prompt and padding tokens do
not add loss.

The teacher and student must have the same token-to-ID mapping. The precompute
stage checks this condition. The training stage checks the saved tokenizer hash
before it loads the student model.

## Teacher-target cache

The cache uses checksummed `safetensors` shards. It stores the tokenized fixed
sequences and the teacher top-20 token IDs and log probabilities for completion
positions. It does not store full-vocabulary logits.

The `metadata.json` file identifies the teacher, tokenizer, dataset selection,
prompt template, maximum length, and top-K value. Student training stops if the
cache does not match `config.yaml`.

The default cache directory is:

```text
teacher_targets/qwen3.5-4b-bf16-hendrycks-math-7500-top20
```

The repository ignores `teacher_targets/`. Git does not upload this data.
Precomputation continues from the last complete shard after an interruption.

## Run on RunPod

Use one NVIDIA A40 with 48 GB of VRAM and a current PyTorch image:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

A separate virtual environment is not necessary in a temporary RunPod
container.

Add the needed keys to `.env`:

```dotenv
WANDB_API_KEY=your_key
HF_TOKEN=your_optional_hugging_face_token
```

Check both commands without model or dataset downloads:

```bash
python precompute_teacher_targets.py --config config.yaml --check-config
python train_off_policy.py --config config.yaml --check-config
```

First, precompute and save the teacher targets:

```bash
python precompute_teacher_targets.py --config config.yaml
```

Then train the student from the saved targets:

```bash
python train_off_policy.py --config config.yaml
```

To keep the cache on a RunPod network volume, use the same absolute path for
both commands:

```bash
python precompute_teacher_targets.py \
  --config config.yaml \
  --output-dir /workspace/opd-cache/qwen3.5-4b-bf16-hendrycks-math-7500-top20

python train_off_policy.py \
  --config config.yaml \
  --teacher-targets /workspace/opd-cache/qwen3.5-4b-bf16-hendrycks-math-7500-top20
```

Use the actual mount path for your RunPod volume. The Pod storage page shows
this path.

`resume_from_checkpoint: auto` resumes from the newest checkpoint in the
off-policy output directory.

## Outputs

The final LoRA adapter is in
`runs/qwen3.5-0.8b-base-math-off-policy-10k/final_adapter`.

Evaluation records are in the run's `evals` directory. Training and evaluation
metrics are also sent to the W&B project in `config.yaml`.

Run local tests with:

```bash
python -m unittest discover -s tests -v
```

## Stand-alone evaluation

Run `Qwen/Qwen3.5-0.8B-Base` on the balanced 50-problem MATH-500 subset with
batched Transformers inference:

```bash
python -m pip install -r requirements-eval.txt
python hendrycks_math.py
```

The command uses greedy decoding, a 2,048-token generation limit, an 8,192-token
total context limit, and the same strict `math-verify` grader as the training
evaluations. The grader uses only the final `\\boxed{}` answer. It does not grade
text in the reasoning field.

The command writes resumable records and a summary to
`runs/qwen3.5-0.8b-base-math500-balanced50`. Each new record has separate
`reasoning` and `response` fields. Records from the old grading method do not
count as completed records and run again.

The command also logs progress and the final result to the `opd` W&B project.
Add `WANDB_API_KEY` to `.env` before the run. Use `--wandb-project`,
`--wandb-entity`, and `--wandb-run-name` to change the W&B destination. Use
`--wandb-mode offline` if the run has no W&B network connection.

Thinking mode is disabled by default. Add `--enable-thinking` only for a model
with a thinking-mode chat template. In thinking mode, the default sampling
values are `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, and
`presence_penalty=1.5`. The generation limit stays at 2,048 tokens. Command-line
sampling options override these defaults.
