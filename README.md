# GEM-pRF Functions cluster runs

This folder contains the frozen inputs and runner for continuing FunctionGemma training from step 4847 on one NVIDIA V100. Start with the pilot on the allocated GPU. The Linux CUDA path requires that pilot before either comparison arm can train.

| File or folder | Purpose |
| --- | --- |
| `experiments.json` | Architecture, training recipe, dataset identities and hashes, experiment IDs, and comparison criteria. |
| `run.py` | The entry point for training, evaluation, and reports. |
| `model.py`, `ledger.py` | The PyTorch model and checkpoint code, and the run log. |
| `evaluate.mjs`, `evaluation/` | The DEV700 runner and the original GEM-pRF evaluator modules. |
| `assets/` | Base weights, tokenizer, starting checkpoint, one shared TRAIN file, DEV700, and evaluator contracts. |
| `runs/` | Local results, checkpoints, and observations. Git ignores this folder. |

## Set up once

Use Linux x86-64, Python 3.13, Node.js 22 or later, Git LFS, and a V100 allocation. Substitute your private repository URL in the clone command.

```sh
git lfs install
git clone https://github.com/YOUR_ORG/gemprf-cluster.git
cd gemprf-cluster
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
```

The repository attributes use Git LFS for weights and TRAIN. Use and distribution of the Gemma weights and derivatives are subject to the restrictions in section 3.2 of the bundled [Gemma Terms of Use](GEMMA-TERMS.md). The model changes are listed in [NOTICE](NOTICE).

## Run on the GPU

```sh
python run.py train pilot-4847-fp32
python run.py test baseline-4847
python run.py train replay-control
python run.py test replay-control
python run.py train replay-treatment
python run.py test replay-treatment
python run.py report
```

The pilot runs one FP32 update using the two longest examples and verifies the saved checkpoint by reloading it. It records time and peak allocated GPU memory. The first FP16 pilot produced a nonfinite loss. Use `pilot-4847-fp32` for the new runtime and keep the original run log for comparison. For interrupted training with the same runtime, rerun the command with `--resume`. Checkpoints are saved every ten updates and at the final update. Use `--run-id NAME` for a new replicate and the same ID for evaluation. To use a pilot with a custom run ID, pass `--pilot-run-id NAME` when training either arm.

## Compare runs

Both arms contain 760 examples in fixed order and run 380 updates. They share 668 examples. The treatment replaces 92 old wordings with independently reviewed new wordings matched by function calls and typed arguments. Both arms cover 97 functions and include 84 outside-scope cases and 40 clarifications. The eligible matched additions were all standalone examples. This experiment measures wording changes with replay.

Each TRAIN line stores one example. `prompt_tokens` marks where the supervised answer starts in `input_ids`. `positions` gives the example's index in each arm that uses it.

`test` runs the 700-case development suite. Session IDs are fixed across arms. The evaluator uses the model's actual conversation history. Prefixes that need independent review remain visible as unresolved. Critical counts are preliminary until independent review. Compare full supported answers and state, whole dialogues, outside-scope responses, critical errors, and unresolved reviews separately.

Each run writes `run.json` with code, architecture, device, checkpoint, TRAIN and DEV identities. `events.jsonl` records progress. `result.json` binds the measured DEV scores to the exact checkpoint and observation log. `report` emits these identities and metrics together as JSON for comparison. Use the same `--runs` directory for jobs that share an execution lock.
