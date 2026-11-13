# GEM-pRF Functions cluster runs

This repository contains frozen inputs and a runner for continuing FunctionGemma training from step 4847 on one NVIDIA V100. Both experiment suites require a successful FP32 pilot before training.

| File or folder | Purpose |
| --- | --- |
| `experiments.json` | Architecture, training recipe, dataset identities and hashes, experiment IDs, and comparison criteria. |
| `experiments-volume.json` | Frozen TRAIN volume comparison and fresh development suite. |
| `run.py` | The entry point for training, evaluation, and reports. |
| `model.py`, `ledger.py` | The PyTorch model and checkpoint code, and the run log. |
| `evaluate.mjs`, `evaluation/` | Development scoring against actual conversation history. |
| `assets/` | Base weights, tokenizer, starting checkpoint, frozen datasets, and evaluator contracts. |
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

## Compare TRAIN volume

Use the `train-volume-v1` branch and keep the checkout fixed until both arms finish. Run these commands from the repository in your existing Python environment. Use the same `runs/` directory for all jobs sharing the GPU. From an empty run directory, the free-space check needs about 684 MiB for the first round or 782 MiB for both rounds, including a 512 MiB reserve. The runner checks space again when saving checkpoints and writing evaluation observations.

```sh
git lfs pull
python run.py --suite volume check
python run.py compare
```

`compare` runs a two-update pilot on the longest inputs and longest target, reloads the pilot checkpoint, scores the original 4847, then trains and evaluates both arms sequentially. Both arms start from the original 4847. The pilot's updates are discarded. The first round compares 1,280 and 1,520 distinct cases after 760 updates per arm. Both arms receive 1,520 presentations with the same capability quotas. The difference in dose is 61 input tokens and four supervised tokens per arm. The wider arm spends 240 draws on distinct matched programs where the smaller arm repeats cases.

After checking the first-round results, you can repeat the frozen order in each arm to reach 1,520 updates per arm.

```sh
python run.py compare --rounds 2
python run.py report
```

The runner saves the exact checkpoint and the DEV observation log for each milestone. The paired results are in `runs/train-volume-v1/comparison-760.json` and `comparison-1520.json` inside the same directory. Supported answers, outside-scope responses, and whole dialogues have separate paired counts. DEV155 includes 51 standalone supported questions, 16 complete five-turn dialogues, and 24 outside-scope questions. The v2 dialogue contract can replay the displayed answer or error, retry a failed exchange, and clarify an ambiguous reference. Score differences are preliminary until independent review of disputed answers.

Reissue the same command after an interruption to continue from its last saved checkpoint. A comparison holds one execution lock across all stages. Its time budget defaults to 36 elapsed hours from first start, including evaluations and interruptions. You can set a smaller budget at first start with `--hours`. Before each round the runner estimates whether both arms and both DEV checks fit from pilot and baseline timings, with a safety margin. Use the cluster allocation for a hard walltime limit. Keep one checkout and shared run directory during execution.

## Earlier wording comparison

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

### Earlier experiment design

Both arms contain 760 examples in fixed order and run 380 updates. They share 668 examples. The treatment replaces 92 old wordings with independently reviewed new wordings matched by function calls and typed arguments. Both arms cover 97 functions and include 84 outside-scope cases and 40 clarifications. The eligible matched additions were all standalone examples. This experiment measures wording changes with replay.

Each TRAIN line stores one example. `prompt_tokens` marks where the supervised answer starts in `input_ids`. `positions` gives the example's index in each arm that uses it.

`test` runs the 700-case development suite. Session IDs are fixed across arms. The evaluator uses the model's actual conversation history. Prefixes that need independent review remain visible as unresolved. Critical counts are preliminary until independent review. Compare full supported answers and state, whole dialogues, outside-scope responses, critical errors, and unresolved reviews separately.

Each run writes `run.json` with code, architecture, device, checkpoint, TRAIN and DEV identities. `events.jsonl` records progress. `result.json` binds the measured DEV scores to the exact checkpoint and observation log. `report` emits these identities and metrics together as JSON for comparison. Use the same `--runs` directory for jobs that share an execution lock.
