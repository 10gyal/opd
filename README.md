# Off-policy distillation on MATH

This project uses `train_off_policy.py` for top-K soft-target cross-entropy on
fixed MATH solutions.

The student is `Qwen/Qwen3.5-2B-Base`. The off-policy teacher is
`Qwen/Qwen3.5-9B-Base`. There are no student rollouts.

## Experiment

- 512 level 3-5 examples from `DigitalLearningGmbH/MATH-lighteval`
- 2,048-token training limit
- Effective batch size 32
- 16 optimizer steps
- Evaluation after steps 8 and 16
- 128 held-out examples from `HuggingFaceH4/MATH-500`
- 1,024 generated tokens per evaluation example
- LoRA rank 32
- Teacher top-20 probabilities for off-policy distillation

The teacher uses 8-bit weight loading by default. This reduces GPU memory use on
one 48 GB A40. Set `teacher.quantization: null` for bfloat16 teacher weights. A
bfloat16 teacher gives more exact soft targets but uses approximately twice the
teacher weight memory.

## Loss

For each fixed solution token, the teacher produces its 20 highest-probability
tokens. Their probabilities are normalized over the selected set. The student
minimizes cross-entropy against these soft targets. Prompt and padding tokens do
not add loss.

The teacher and student must have the same token-to-ID mapping. The trainer
checks this condition before training.

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

Check the configuration without model or dataset downloads:

```bash
python train_off_policy.py --config config.yaml --check-config
```

Run off-policy distillation:

```bash
python train_off_policy.py --config config.yaml
```

`resume_from_checkpoint: auto` resumes from the newest checkpoint in the
off-policy output directory.

## Outputs

The final LoRA adapter is in
`runs/qwen3.5-2b-base-math-off-policy/final_adapter`.

Evaluation records are in the run's `evals` directory. Training and evaluation
metrics are also sent to the W&B project in `config.yaml`.

Run local tests with:

```bash
python -m unittest discover -s tests -v
```

## Stand-alone evaluation

Run `Qwen/Qwen3.5-0.8B-Base` on the 500-problem test split of the Hendrycks
MATH benchmark with batched Transformers inference:

```bash
python -m pip install -r requirements-eval.txt
python hendrycks_math.py
```

The command uses greedy decoding, a 4,096-token generation limit, an 8,192-token
total context limit, and the same
`math-verify` grader as the training evaluations. It writes resumable records
and a summary to `runs/qwen3.5-0.8b-base-hendrycks-math`. It also logs progress
and the final result to the `opd` W&B project. Add `WANDB_API_KEY` to `.env`
before the run. Use `--wandb-project`, `--wandb-entity`, and `--wandb-run-name`
to change the W&B destination. Use `--wandb-mode offline` if the run has no
W&B network connection. Thinking mode is disabled by default. Add
`--enable-thinking` only for a model with a thinking-mode chat template.
