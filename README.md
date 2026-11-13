# GEM-pRF Functions cluster runs

This repository contains frozen inputs and a runner for continuing FunctionGemma training from step 4847 on NVIDIA V100 GPUs. Every training course requires a successful FP32 pilot. The full course trains one model on four GPUs; the earlier volume experiment compares two datasets.

| File or folder | Purpose |
| --- | --- |
| `experiments.json` | Architecture, training recipe, dataset identities and hashes, experiment IDs, and comparison criteria. |
| `experiments-volume.json` | Frozen TRAIN volume comparison and fresh development suite. |
| `experiments-full.json` | Full TRAIN course: batch four, three epochs, resource limits and checkpoint selection. |
| `run.py` | The entry point for training, evaluation, and reports. |
| `model.py`, `ledger.py` | The PyTorch model and checkpoint code, and the run log. |
| `evaluate.mjs`, `evaluation/` | Development scoring against actual conversation history. |
| `assets/` | Base weights, tokenizer, starting checkpoint, frozen datasets, and evaluator contracts. |
| `runs/` | Local results, checkpoints, and observations. Git ignores this folder. |

## Set up once

Use Linux x86-64, Python 3.13, Node.js 22 or later, Git LFS, and a V100 allocation.

```sh
git lfs install
git clone https://github.com/wassilysavin/for-mu-cluster.git
cd for-mu-cluster
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
```

The repository attributes use Git LFS for weights and TRAIN. Use and distribution of the Gemma weights and derivatives are subject to the restrictions in section 3.2 of the bundled [Gemma Terms of Use](GEMMA-TERMS.md). The model changes are listed in [NOTICE](NOTICE).

## Full TRAIN on four GPUs

With earlier jobs stopped, use your existing environment and checkout:

```sh
git fetch origin
git switch full-train-v2
git pull --ff-only
git lfs pull
python run.py --suite full check
python run.py full --gpus 4
```

This is one model trained from the bundled **4847 checkpoint**, not from a remote intermediate run. The frozen corpus contains 2,977 distinct examples: 2,941 retained examples and 36 reviewed recovery examples using the actual-dialogue v2 contract. Each epoch contains 3,144 presentations, including 120 outside-scope and 47 clarification repeats. Its fixed order repeats for three epochs. There is no random shuffle or mutable data generation on the cluster.

Five retained retry targets are corrected for the v2 failed-exchange contract, and five later histories are rebuilt coherently from those corrections. The earlier dataset files remain unchanged. Each line of `assets/train-full-v2.jsonl` is one case: `question` is readable user text, `calls` is its supervised function program, and `history_origin` identifies the provenance of its context. `input_ids` contains tokenizer IDs, `prompt_tokens` marks the answer boundary, and `positions.full` / `positions.pilot` specify exactly when the case is presented. Constructed recovery examples are identified as constructed contexts; they are not represented as fresh model-error observations.

Visible GPUs 0–3 each process one example, then their token-weighted gradients are summed for one Adafactor update. The global batch is **four**, intentionally different from the earlier volume experiment's batch of two. The learning rate remains 1e-4. Each epoch has 786 updates; epoch checkpoints are steps 5633, 6419 and 7205. This course therefore cannot isolate the effect of TRAIN size by comparison with the old batch-two runs.

The runner first audits two extreme four-example batches against an independent sequential reference, repeats each reference update from identical state, and verifies checkpoint save/reload. Strict deterministic kernels, complete gradient capture and the numerical gates are mandatory. Pilot updates do not initialize the course. Next it evaluates the original 4847 on all 155 DEV cases, then trains and evaluates each full epoch. Evaluation preserves complete five-turn dialogues and runs on the primary GPU. TEST is absent from the course and remains closed.

One shared execution lease covers the entire course. Frozen data, code, runtime and GPU identities are recorded. Checkpoints include optimizer state and the schedule cursor, and are saved every ten updates and at each epoch boundary. Reissue the same command after interruption; do not pull new code during a run. The command resumes its original identity, elapsed deadline and optimizer budget. Repeated or interrupted attempts consume the same budget; the maximum is 3,246 charged updates including pilot reference checks. The elapsed limit is 36 hours from the first start, including disconnected time and evaluations. A smaller `--hours` can be set only when first creating a course. The cluster allocation should also impose its hard walltime limit.

Before each epoch, the runner checks remaining disk and estimated training-plus-DEV time. Its first estimate uses the slowest measured pilot update; later estimates use completed course updates with a safety margin. If another complete epoch cannot fit, it reports the checkpoints already evaluated. Inputs are shared, without additional dataset copies. Output-space admission counts the pilot checkpoint, three epoch checkpoints, one temporary checkpoint, four DEV logs and a 512 MiB free-space reserve. Existing files and other folders are not deleted.

The default controller directory is `runs/full-train-v2-4gpu/`. Its `course.json` fixes the clock and experiment identities; `optimizer-budget.json` records charged updates. `selection-epoch-N.json` records each provisional choice. Final `result.json` contains baseline and epoch scores, the selected checkpoint and any budget-related stop. Individual pilot, baseline and course logs are in the adjacent folders named with the same prefix.

Selection requires strictly more complete supported answers while preserving whole-dialogue and outside-scope success, without increasing unresolved reviews or allowing detected critical errors. Equal supported scores keep the earlier checkpoint. This is provisional DEV selection; it is not product acceptance or a guarantee of quality.

To survive an SSH disconnect, open a tmux session from the repository with your environment active:

```sh
tmux new -s gemprf-full
```

Inside that session, with the same environment active, start the course:

```sh
python -u run.py full --gpus 4
```

Detach with `Ctrl-B`, then `D`. Reconnect after logging in with `tmux attach -t gemprf-full`. Start the course only once; reconnect to its existing session to inspect progress.

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

### Four GPUs

After the previous command has exited, update the checkout and use four V100 GPUs from your cluster allocation:

```sh
git pull --ff-only
python run.py compare --gpus 4
```

The default run ID is `train-volume-v1-4gpu-v2`, separate from the one-GPU comparison. It starts both arms from 4847 and preserves all existing results. The earlier `train-volume-v1-4gpu` pilot records remain separate. New code or GPU layout cannot silently resume an old run identity. Do not update the checkout during an active comparison.

Narrow uses visible GPUs 0 and 1; wide uses 2 and 3. Each pair processes the same two examples per update concurrently, sums gradients normalized by their combined supervised-token count, and applies one Adafactor update. Batch size, learning rate, example order, and checkpoint format remain unchanged. `CUDA_VISIBLE_DEVICES`, if set by your allocation, determines these logical indices. The recorded identity includes the selected devices and execution mode.

Each pair must first match two independent sequential reference updates for loss, normalized gradients, adapter weights, and optimizer state, then save and reload its checkpoint. Each reference update is also repeated from its identical saved in-memory state to check repeatability. The original reference trajectory is restored afterwards; it is not aligned to the paired trajectory. All differences are recorded in a `pilot_numerical_audit` event before a failed comparison stops execution. Four-GPU mode enables strict deterministic PyTorch algorithms and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization (an existing `:16:8` setting is also accepted). The execution identity records this policy. Unsupported deterministic operations stop the pilot. Loss, weight, and optimizer tolerances are unchanged from v1; the additional gradient check also limits relative L2 error to 1e-5. This hardware pilot is required; CPU fixture checks are not a substitute. After a fresh baseline DEV evaluation, the two training arms run concurrently. Each arm's DEV evaluation runs on its primary GPU with complete dialogues in order, so all four GPUs are used for training, while baseline and evaluation stages use fewer GPUs. Speedup is measured by the pilot rather than assumed to be fourfold.

The first round needs about 718 MiB free, including the 512 MiB reserve, two pilot checkpoints, and two simultaneous temporary checkpoints. Both rounds from scratch need about 815 MiB. Inputs are shared without extra copies. Each worker inherits the coordinator's execution lock and stops if the coordinator disconnects; a worker failure stops its sibling and keeps saved checkpoints. Reissue the same command to recover. An unverified pilot restarts from 4847, and completed evaluations are reused.

Results are written to `runs/train-volume-v1-4gpu-v2/comparison-760.json`. For the optional second round, use `python run.py compare --gpus 4 --rounds 2` with the same checkout and run ID. The 36-hour budget remains measured from the comparison's first start. The time gate uses the slower parallel arm, including its DEV evaluation.

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
