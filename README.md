# GEM-pRF Functions cluster runs

This repository contains frozen FunctionGemma experiments and a separate post-run diagnostic for NVIDIA V100 GPUs. The diagnostic measures TRAIN fit and compares the completed history experiment with a pinned Qwen3-1.7B reference. Earlier experiments and their inputs remain available. Every training course requires a successful FP32 pilot.

## Compare paraphrases and semantic contrasts on four V100s

From your existing checkout of `train-fit-reference-v1` with the project environment active, run this in your cluster tmux session:

```sh
./cluster git pull --ff-only && ./cluster python -u transfer_run.py --gpus 4
```

Both arms start from the saved FunctionGemma treatment checkpoint at step 7991 with its FP32 optimizer state. Each trains for 160 updates of four examples, using four V100s together. The runner completes the arms sequentially under one lease. It checks the saved cluster artifacts and disk allowance, runs a numerical pilot, and resumes recorded progress when the same command is repeated. All transfer artifacts are inside this checkout. The combined additional disk budget is 3 GB, including the existing reference-model cache and reserve.

The two TRAIN files use the same 160 dialogue roots and the same overall distribution of wording and intended actions. The ordinary arm pairs two paraphrases of one request with a history. The contrast arm pairs two different requests with that history. For example, a user can ask to quote the displayed failure or retry the underlying question. Each arm has 320 variable presentations and 320 common replay examples. Half of the variable presentations also coincide across arms. Sixteen roots use saved student histories. The runner reconstructs the other 144 and verifies native targets and prompt/target token balance before training.

The new DEV contains 161 primary turns, including 24 four-turn dialogues, and a separate 72-case panel with controlled histories. Each candidate generates its own dialogue history. The report compares full delivered answers and dialogue state, whole dialogues, outside-scope handling, critical errors and unresolved reviews. It also checks the frozen DEV155 and DEV1250 regressions. The first seed is 1729. A gain of at least five percentage points on supported primary turns, a positive lower family-bootstrap confidence bound, more complete dialogues and passing regression guards trigger the confirmatory pair with seed 2718. The terminal prints the effect and interval. The complete comparison is saved in `runs/semantic-transfer-v1/result.json`.

The dataset declarations are in `assets/transfer-train-*.jsonl` and `assets/transfer-dev-v1.json`. The CPU preflight is `./cluster python transfer_run.py --check`. The new DEV is public and diagnostic. The independent product TEST remains closed.

## Inspect saved FunctionGemma errors on DEV1250

From the existing `train-fit-reference-v1` checkout on the cluster, run:

```sh
./cluster git pull --ff-only && ./cluster python inspect_student_errors.py
```

The report checks the completed baseline, control and treatment against their saved receipts and replays their recorded outputs on the CPU. It prints panel totals, recorded failure categories, differences between calls, and case-level gains and losses. Error details default to supported closed-loop cases. Use `--panel fixed_prefix` or `--panel challenge` for the other panels. Counts of call differences can overlap. Each model saw its own conversation history. The command prints to the terminal and requires the original cluster run files.

## Recheck Qwen format on the same 32 TRAIN cases

After `TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED`, run this in the existing environment and checkout:

```sh
./cluster git pull --ff-only && ./cluster python -u qualify_reference.py --gpus 4
```

This runs **only the original 32 qualification cases**, split across four V100 GPUs. It performs no training, student evaluation or DEV generation. It uses the existing pinned Qwen snapshot offline; no weights or dependencies are downloaded. Results are separate under `runs/train-fit-reference-v1/revisions/qwen-hermes-v1/result.json`. The original diagnostic's 3 GB disk account includes this child directory and its existing results/cache; the account is not reset. The shared execution lease and private temporary paths remain in effect.

The adapter requests Qwen's documented [Hermes JSON tool-call format](https://qwen.readthedocs.io/en/latest/framework/function_call.html), using the official non-thinking chat wrapper and the complete original compact function contract. It does not expand that contract into a duplicated per-tool `tools=` payload. Only two transport instruction lines and the output-format instructions change; the question, history, function meanings and all argument constraints are retained. Greedy decoding, FP16 weights, context and completion limits remain those of the previous reference recipe; this is not a claim of optimal Qwen quality. Each worker explicitly enables the recorded deterministic numerical policy.

Every original response and token ID is saved. Only a complete, valid Hermes batch ending at the model's EOS is serialized into the executor's native format. Names, arguments, types and order remain model-selected; prose, malformed JSON, duplicate keys and unfinished batches are failures, not repaired outputs. Admission still requires at least **30/32 valid batches and one complete supported answer**. Passing this check does not start any further experiment or establish domain quality or product acceptance.

Use `./cluster python -u qualify_reference.py --gpus 4` to resume after interruption, without pulling new code. Saved responses are verified and reused; a completed run only verifies and displays its report without loading a model. Run inside your existing private tmux session if you may disconnect. Offline checks: `./cluster python -m unittest test_qwen_transport test_qualify_reference test_inspect_reference`.

To explain the saved nonpassing answers without running Qwen again:

```sh
./cluster git pull --ff-only && ./cluster python inspect_qwen_failures.py
```

The reporter authenticates all 32 saved observations and reproduces their deterministic executor scores on CPU. It prints recorded failures, critical and unresolved counts, structural differences in calls/arguments, and four representative examples. Comparisons consider every frozen allowed program, preserve duplicate calls and distinguish argument types. Structural signals can overlap and are measured against a closest allowed program; ambiguous alignments are reported. These signals are debugging clues, not independent adjudication of unresolved answers or proof that a scientific label is correct. Passed semantic equivalents are not relabeled as failures. `--all` lists one short line for every nonpassing case; `--examples 0` prints only the aggregate report. No model, CUDA, weight download or saved-result write is involved. Tests: `./cluster python -m unittest test_inspect_qwen_failures`.

For the reported **one missing contract, four unresolved answers and eight argument errors**:

```sh
./cluster git pull --ff-only && ./cluster python inspect_qwen_cases.py
```

This checks all 32 saved identities and receipts before inspecting the 13 selected cases. It prints their exact questions, saved calls, relevant history excerpts, and the compiler/executor's actual error messages. Dynamic argument schemas use the original conversation state. A missing exact source binding is distinguished from a schema violation; existing binding examples do not establish the intended answer. Each unresolved answer is compared with every frozen allowed program, separating delivered-text differences from dialogue-state differences. Reviews remain unresolved and original scores stay unchanged. This is CPU replay of saved calls, with no model, GPU, download, training or new DEV generation.

The short output explicitly marks clipped context. `--full` prints complete selected histories and diagnostic JSON, including complete actual/expected states for unresolved cases. It only writes to the terminal. Run in the same private checkout and environment; no additional dependencies or assets are needed. Offline checks: `./cluster python -m unittest test_inspect_qwen_cases` and `./cluster node --test test_inspect_qwen_cases.mjs`.

To copy the **three remaining disputed cases together** after discussing the first review:

```sh
./cluster git pull --ff-only && ./cluster python inspect_qwen_cases.py --remaining-reviews
```

This compact view excludes only the already displayed `control:history-v3-train-error_and_success-18-fresh_after_success-control` case. It does not resolve or remove any of the four saved review flags. Each remaining case shows its question, Qwen calls, the first frozen allowed program, and display/state/both match counts across **all** alternatives. The complete compact exchange map gives `[exchange ID, outcome, error, semantic turn ID, answer IDs, response-part kinds]` for every actual prior exchange, including failures and clarifications. Exchange numbers are never substituted for semantic turn IDs; zero answer objects alone do not establish that no scientific text was displayed. Reply excerpts surround the first mismatch against alternative zero; state paths are explicitly sampled. Long text fields are marked as clipped; complete evidence remains available with `--full`. Original numbering is retained. Authentication and CPU replay are unchanged, and no model is run.

## Finish the stopped current comparison

The original `history --repair-parity` command retains its 36-hour budget and returns an existing `budget_stopped` result. The user authorized completing both current arms and their checks regardless of duration on 2026-11-16. Once that coordinator has exited, use the separate continuation below inside the same personal checkout and original Python environment:

```sh
. ../.venv/bin/activate
./cluster git fetch origin &&
./cluster git switch train-fit-reference-v1 &&
./cluster git pull --ff-only &&
./cluster python -u finish_history.py --gpus 4
```

Run this inside an existing tmux session. If outside tmux, first enter one with `./cluster tmux new -A -s gemprf-history`, then run the commands there. This uses the private socket in the checkout. Do not nest tmux sessions or run two coordinators. The shared execution lease rejects concurrent model jobs.

This continuation authenticates the original numerical PASS, collected TRAIN, worker identities, schedules, checkpoints and saved DEV prefixes. It finishes only **control-1729 and treatment-1729**: 786 updates per arm from 7205 to 7991, known DEV155 at 7598, and all 1,250 fresh DEV observations at 7991. A completed control training checkpoint and its known DEV are reused. Only missing fresh DEV replies are generated; treatment resumes from its latest saved checkpoint if present. Four GPUs train each arm; the original evaluator still uses the primary GPU. No new model or dependency download is needed.

Only the elapsed-time cutoff is removed. Original model/training/evaluation files, dataset hashes, optimizer recipe, 3,246 cumulative update ceiling, disk admission and 512 MiB reserve remain intact. The original `budget.json`, `result.json`, manifests and existing stage receipts are preserved. Pending worker checkpoints and evaluation logs advance in their existing paths; the new manifest, inherited work account and stage receipts are under:

```text
runs/actual-history-v3-4gpu/revisions/rope-alignment-v1/recovery/checkpoint-io-v1/completion-v1/
```

Repeat `./cluster python -u finish_history.py --gpus 4` after an interruption without changing files or resetting accounts. Its final `result.json` compares the full first pair using the original gates; unresolved reviews remain unresolved. It does not start seed 2718 or establish product acceptance. Conservative elapsed accounting can include an uncertain offline interval, so it is not GPU compute time. The TRAIN-fit diagnostic below automatically uses and authenticates this completed continuation; an incomplete continuation blocks diagnostic admission.

## TRAIN fit and stronger reference

To inspect a completed reference admission, including `TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED`, run:

```sh
./cluster python inspect_reference.py
```

The reporter reads the saved identities, qualification receipt and all 32 responses, verifies their hashes and counts, then prints failed conditions and two bounded response excerpts. It distinguishes insufficient valid native formats from zero complete supported answers. It does not load models, access the network, rerun evaluations or change saved results. A newer reporting-only Git commit is allowed; the original saved experiment identity remains authoritative. Use `--examples 0` for counts only or `--examples 4` for more examples.

Run this diagnostic **after the actual-history comparison has finished**, using the continuation above if it stopped on time. Keep the checkout unchanged while a coordinator is running. The diagnostic requires a saved terminal result containing the baseline and both complete seed-1729 arms; it refuses partial pairs and an active execution lease before downloading any model. The diagnostic itself does not resume, retrain or modify the history experiment.

Inside your personal `vsavin/for-mu-cluster` checkout, with the existing environment active:

```sh
./cluster git fetch origin &&
./cluster git switch train-fit-reference-v1 &&
./cluster git pull --ff-only &&
./cluster git lfs pull &&
./cluster python -m pip install --no-cache-dir --no-compile --no-deps --only-binary=:all: gguf==0.17.1 accelerate==1.10.1 psutil==7.0.0 &&
./cluster python diagnose.py --check &&
./cluster python -u diagnose.py --gpus 4
```

Run the final command inside your existing tmux session. If you need a new one, start `./cluster tmux new -s gemprf-diagnostic`, then run `./cluster python -u diagnose.py --gpus 4` inside it. Reconnect with `./cluster tmux attach -t gemprf-diagnostic`. After an interruption, repeat the same diagnostic command with the same code and run directory. Do not reset source budgets, remove checkpoints, or create a second coordinator.

**Storage:** this revision replaces the earlier 8B proposal. It uses a **3,000,000,000-byte additional disk budget**, including new diagnostic results, one shared reference snapshot, temporary files/cache growth, a 512 MiB free-space reserve and 64 MiB allowance for the Git/LFS sidecar and small dependency. Existing environments, source runs and frozen assets are retained separately and do not have to fit into that additional allowance. Have 3 GB free before the branch update. The eight pinned reference files total **1,850,319,396 bytes**; weights alone are 1,834,426,016 bytes. GGUF expands into RAM, with no FP16 weight copy on disk. Four workers share those files. No existing folders or caches are cleaned.

Before downloading, the coordinator admits the missing pinned bytes plus 384 MiB for outputs and cache metadata. Every controlled result write is serialized under one disk lock and checked against the cumulative budget and current free space. It counts allocated bytes, deduplicates hardlinks, includes private cache growth, and never treats deleting an old cache as new budget. A larger-than-expected result or external loss of free space stops safely before the next record; no cases are silently discarded. The budget bounds storage; completion still depends on actual output sizes and shared-filesystem availability. This is application-level accounting, not an operating-system quota on other processes.

**Product limit:** the browser model still must fit within **200 MB**. Qwen is a temporary diagnostic reference, not an application artifact. Higher reference quality would guide experiments; it would not prove that the same quality can be compressed into 200 MB.

**Hardware:** four allocated V100 32 GB GPUs, with at least 12 GiB free per GPU. The existing pinned `requirements.txt` stays unchanged; the extra small dependencies are `gguf==0.17.1`, `accelerate==1.10.1` and `psutil==7.0.0`, installed from binary wheels without a dependency upgrade or build. Each GPU holds an independent inference replica. Model loading is staggered to reduce host-memory pressure. Cases are distributed across GPUs; all turns of one DEV dialogue stay on one GPU. This is parallel inference, not distributed training.

The diagnostic performs:

1. Authenticate the latest completed history recovery, its numerical PASS receipt, checkpoint lineage, frozen schedules and complete DEV observations. Measure baseline 7205 and both seed-1729 checkpoints 7991; do not select a winner or cherry-pick the better seed.
2. Restore the exact fixed TRAIN inputs. The panel contains 1,572 common examples and 480 examples from each history arm: 2,532 distinct cases. The 720 historical common contexts are supplied in a checked sidecar, reconstructed from original authoring records and matched to the original tokenized prompts. Existing TRAIN is unchanged.
3. Admit the reference transport on 32 predetermined TRAIN-only cases, including the longest native-token prefix. At least 30 must produce parseable native batches and at least one supported case must pass fully. A failed format gate withholds reference DEV but still measures all three students' TRAIN fit and saves the failed reference evidence. This is a transport check, not proof of strong domain competence.
4. Generate answers for all 2,532 fixed cases with each student and Qwen. Score full delivered answer plus state, literal native exact match, canonical call match, errors and unresolved cases separately. Legitimate complete equivalent programs are accepted by the semantic rubric. Frozen TRAIN labels remain the reference; this does not independently establish their scientific truth.
5. Evaluate Qwen on the unchanged DEV1250, retaining its own actual histories. Reuse the students' authenticated, complete source DEV observations with the unchanged executor/rubric, rather than run their evaluations again. Fixed-prefix inputs are identical; closed-loop inputs naturally differ after each model's own earlier answers. Report panels and whole dialogues separately, with paired pass/fail counts and unresolved cases disclosed.

Each trained arm saw 2,052 distinct examples in 3,144 presentations. Its **own-seen TRAIN fit** excludes the other arm's 480 examples; those are reported as cross-history transfer. Baseline fit covers the 1,572 common members of its earlier full-v2 TRAIN, not all 2,977 original examples. Qwen did not train on this corpus, so its score is reference performance on the TRAIN panel, not memorization fit.

The reference is the official [`Qwen/Qwen3-1.7B-GGUF` Q8_0](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/tree/90862c4b9d2787eaed51d12237eafdfe7c5f6077), pinned to revision `90862c4b9d2787eaed51d12237eafdfe7c5f6077`. Native configuration/tokenizer files are pinned separately to [`Qwen/Qwen3-1.7B@70d244c`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e); no native weight shards are downloaded. Official Transformers 4.57.3 loads the quantized GGUF into FP16 memory. This measures a quantized reference recipe, not equivalence to the original full-precision weights. Non-thinking greedy decoding has a 2,048-token generation limit. FunctionGemma keeps its original 512-token limit; tokenizers differ. Qwen receives the same developer contract and complete JSON question/history, with a mechanical chat-wrapper change and explicit native-call formatting instructions. There is no semantic routing, output repair, reasoning extraction, hidden tool-name conversion or context truncation. Prefill uses 256-token chunks and the official full cache with MATH attention. Context overflow remains a recorded planned failure; OOM and nonfinite logits stop for investigation and preserve completed observations.

Greedy non-thinking inference is a fixed diagnostic recipe; Qwen's model card recommends sampling. A poor reference result may reflect prompting, format adaptation or domain knowledge. It is not a demonstrated ceiling of the compact model. No new training, parameter updates, sealed TEST access or product acceptance occur in this command.

Results are in `runs/train-fit-reference-v1/result.json`. `COMPLETE_DIAGNOSTIC` means both TRAIN fit and reference DEV completed; `TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED` means student TRAIN fit completed but the reference failed its format gate. Neither is product acceptance. Raw generations, original tokenizer IDs, hashes, delivered replies, all scores and per-GPU observations remain beside the report. Shared TRAIN prompts are reconstructed from the immutable tokenized TRAIN file; original before states are stored once. Compact receipts replace repeated states with hashes. Replaying the pinned executor restores full states without another model call. Frozen identities bind resumes to each checkpoint, phase and GPU assignment. Dataset, code, model revision, source results and dependency versions cannot silently change during a resumed diagnostic. The same shared execution lease covers downloads and all model workers; workers exit if their coordinator disappears.

Local checks before publication cover recovered contexts, equivalent programs, complete dialogue sharding, resume integrity, shared disk accounting, compact receipt reconstruction, private paths and a tiny random official-Qwen dense/chunked calculation. Actual reference quality, FP16 behavior and V100 memory fit are measured only when this cluster command runs.

```sh
./cluster python -m unittest test_diagnose test_diagnose_integration test_reference_model
./cluster node --test test_diagnose_data.mjs test_diagnose_dev.mjs
```

| File or folder | Purpose |
| --- | --- |
| `experiments.json` | Architecture, training recipe, dataset identities and hashes, experiment IDs, and comparison criteria. |
| `experiments-volume.json` | Frozen TRAIN volume comparison and fresh development suite. |
| `experiments-full.json` | Full TRAIN course: batch four, three epochs, resource limits and checkpoint selection. |
| `experiments-history.json` | Current actual-history comparison from checkpoint 7205, including fresh DEV and matched schedules. |
| `run.py` | The entry point for training, evaluation, and reports. |
| `model.py`, `ledger.py` | The PyTorch model and checkpoint code, and the run log. |
| `evaluate.mjs`, `evaluation/` | Development scoring against actual conversation history. |
| `assets/` | Base weights, tokenizer, starting checkpoint, frozen datasets, and evaluator contracts. |
| `runs/` | Local results, checkpoints, and observations. Git ignores this folder. |
| `cluster`, `private_runtime.py` | Launch commands with temporary files and dependency caches confined to `.runtime/` in this checkout. |
| `diagnose.py`, `diagnose-data.mjs`, `diagnose-dev.mjs` | Post-run TRAIN-fit measurement and four-GPU reference comparison. |
| `finish_history.py` | Finish the existing seed-1729 pair without an elapsed-time cutoff, preserving its original protocol and records. |
| `inspect_reference.py` | Read a completed reference-admission report and display verified counts and short failed-response examples. |
| `qualify_reference.py`, `qwen_transport.py` | Recheck only the same 32 TRAIN cases using Qwen Hermes output and the existing weights. |
| `inspect_qwen_failures.py` | Explain saved qualification failures and call/argument differences without model inference. |
| `inspect_qwen_cases.py`, `inspect_qwen_cases.mjs` | Trace the 13 selected saved failures, exact schema/binding errors and unresolved answer/state differences. |
| `reference_model.py` | Pinned local Qwen download, transport adapter and bounded-memory inference. |
| `assets/diagnostic-train-contexts.*` | Exact restored TRAIN histories and their provenance; no new training examples. |

## Private temporary files

Keep this checkout inside your personal directory, for example `vsavin/for-mu-cluster`. Use `./cluster` before Python, pip, Node or new tmux commands:

```sh
./cluster
./cluster python test_history.py
./cluster python -m pip install -r requirements.txt
```

The launcher overrides inherited temporary/cache settings before the child process starts. Default Python and Node temporary directories use `.runtime/tmp`; pip, PyTorch, Triton, CUDA and Hugging Face caches use `.runtime/cache`. Atomic checkpoint and receipt staging remains beside the destination inside `runs/`. New tmux sessions use the explicit short socket `.runtime/tmux`. Python bytecode writes are disabled. Scratch directories must be real directories owned by your user and not writable by others; symlinks are rejected. Tests remove only their newly created temporary directories. Existing external caches are neither moved nor deleted. `HOME` and system settings are not changed.

Python entry points also configure these paths, but the launcher is required to cover interpreter startup, pip and other child tools. This configures this workflow; it is not an operating-system sandbox for arbitrary commands. `.runtime/` is excluded from Git.

## Checkpoint cleanup recovery

The latest V100 run passed the exact RoPE controls, short dense/chunked MATH controls and both complete native generations (48 and 43 tokens). It then stopped while cleaning the temporary checkpoint-reload directory with `Errno 39: Directory not empty`. Training and TRAIN-history collection had not started; no complete parity PASS receipt was published.

The checkpoint reader returned CPU tensors sharing the safetensors file mapping. Those tensors could keep the temporary file open after its reader context exited, preventing cleanup on NFS. The reader now copies tensors into owned memory before validation. JSON publication also closes its temporary file before linking or unlinking it. Cleanup failures cannot replace an earlier validation error. These are resource-lifetime corrections; model calculations, weights, optimizer, data and tolerances are unchanged.

Inside the same checkout, active environment and existing tmux session:

```sh
git pull --ff-only &&
./cluster python -u run.py history --gpus 4 --repair-parity
```

The fixed `checkpoint-io-v1` recovery validates the stored cleanup failure and frozen `rope-alignment-v1` evidence before creating its own manifest. A recorded numerical failure, existing parity PASS or started training blocks this recovery. All earlier manifests, failures and checkpoints remain intact. The recovery records the old and new source hashes and shares the original budget and execution lease. Spent time is retained; budget rollback and `--hours` overrides are rejected.

The official reference continues to use MATH, its own DynamicCache and prefill chunks of at most 256 tokens without dropping prompt tokens. The gate checks canonical RoPE components at long positions, repeats short dense/chunked controls and full-prompt logits, compares complete native calls for both the median and longest prompts, and checks loss, gradients and checkpoint reload. The 7205 baseline and both training arms will be evaluated using the same corrected runtime. The existing four-GPU pilot remains mandatory before training.

A new numerical mismatch blocks training and retains its failure report. If time remains, a full-prompt mismatch also captures compact layer differences in the same process. For an ordinary interruption, repeat the same command without changing the checkout. Recovery reports are under `runs/actual-history-v3-4gpu/revisions/rope-alignment-v1/recovery/checkpoint-io-v1/`; the shared account remains the original `budget.json`.

The complete gate is repeated before collection and training. The cleanup correction has local regression tests; the complete recovered V100 run remains unverified. Do not delete the failed run, reset its budget, or restart it under a new ID. All temporary paths remain inside the personal checkout.

The earlier RoPE correction remains: canonical CPU `1 / base**exponent` frequencies replace per-call GPU `base**(-exponent)`. Its previous long-prompt logit discrepancy no longer appeared in the latest V100 log. The original MATH and RoPE failure evidence is retained.

## Current experiment: actual-history-v3

This branch compares how the model learns to address actual dialogue history. It starts from your completed **7205 checkpoint and its optimizer**, already stored under `runs/full-train-v2-4gpu-full-course/`. It never substitutes the earlier 6419 selection or the bundled 4847 checkpoint.

From this repository, inside your existing environment and tmux session:

```sh
git fetch origin
git switch actual-history-v3
git pull --ff-only
./cluster git lfs pull
./cluster python -m pip install -r requirements.txt
./cluster python test_history.py
./cluster python run.py --suite history check
./cluster python -u run.py history --gpus 4
```

If you are already inside tmux, run the command there. Otherwise create a private session with `./cluster tmux new -s gemprf-history` first. Reconnect to that session using `./cluster tmux attach -t gemprf-history`; earlier sessions created without the launcher still use their original socket. For an ordinary interruption, repeat **the same** `./cluster python -u run.py history --gpus 4` command with this checkout unchanged. The existing cleanup failure requires the explicit recovery above; numerical failures require investigation. Fresh runs use the same corrected runtime and bounded MATH reference. One repository lease prevents duplicate model jobs.

The run performs these stages automatically:

1. Verify the real 7205 weights, complete optimizer, original full-course receipts and frozen inputs. Compare the custom runtime with **Transformers 4.57.3 Gemma3**, using identical effective weights, tokenization and declared numerical tolerances. This is an offline reference check; no extra model download occurs. Numerical failure blocks training.
2. Collect 360 TRAIN-only prefixes from 7205, requiring 1,452 generation requests before resumptions. Materialize 480 reviewed continuation policies across 192 scenario families and eight groups. Raw model replies remain in the histories; they never become supervised gold merely because the model generated them. Actual histories are collected here because the 7205 checkpoint is on your cluster.
3. Freeze tokenized schedules. Every update contains two common old TRAIN examples and two history examples. Common replay covers all 97 domain functions. Both seeds use 3,144 presentations per arm (786 updates); prompt and supervised-token totals must each differ by at most 5%, without truncation or dropped cases.
4. Run the four-GPU numerical pilot, evaluate the unchanged 7205 baseline, then train control A and treatment B **sequentially**, each using all four allocated GPUs. Evaluate known DEV155 at update 393 and the fresh panels after update 786. Evaluations and history collection use the primary GPU; training uses all four.
5. Apply the frozen diagnostic gates. Run the second pair from the same 7205 with seed 2718 only if the first pair passes and the complete second pair fits the remaining budget. Unresolved semantic reviews or infrastructure failures prevent automatic promotion.

**Prospective protocol amendment:** strict old-row matching was impossible for the new dialogue scenarios. Both variable halves therefore use the same collected content: A reconstructs the earliest legal, content-preserving target order; B retains the actual order. Address wording and native IDs are remapped together. Invalid-reference errors, conditions and pending requests must retain their meaning. If no earlier order preserves them, the identity pair is retained and counted. This tests the position/order training package, **not** pure history provenance or old-data-only continuation. The 50% common replay remains identical at the same positions. See `assets/history-amendment.json`.

Fresh evaluation has separate denominators: 288 fixed-prefix cases; 898 closed-loop turns (834 supported, 64 outside scope, 128 whole dialogues, including 194 standalone questions covering 97 functions); and 64 additional turns in eight long dialogues. Complete answer plus state is primary. Whole-reply and answer-level quotations admit equivalent complete programs only for the same actual occurrence and scope. Repeated questions link whole scenario families into conservative bootstrap blocks. No fresh DEV output is used to author TRAIN. Sealed TEST is absent and remains closed.

The entire experiment has a 36-hour model-worker budget and at most 3,246 charged optimizer updates, including pilots and repeated work after interrupted checkpoints. It preserves a 512 MiB free-disk reserve, verifies hashes on resume, keeps atomic checkpoints and records actual before/after states. Interrupted phase time is conservatively charged through recovery; the budget does not reset. The larger-model reference and a separate model-based sample-sizing study are deferred. Panel sizes are engineering choices, not a claim of statistical optimality.

Files added for this experiment:

| File | Purpose |
|---|---|
| `experiments-history.json` | Frozen assets, recipe, panel sizes and prospective amendments. |
| `history.py` | Stage orchestration, budget, checkpoint provenance and paired decisions. |
| `history_prepare.py`, `history-data.mjs` | TRAIN-only collection and matched schedule preparation. |
| `history_parity.py` | Independent official-runtime numerical gate. |
| `evaluate-actual.mjs`, `evaluate-known.mjs`, `evaluation/` | Actual-history evaluation and the reviewed old DEV rubric. |
| `assets/history-*`, `assets/actual-history-v3-dev.json` | Readable scenario specifications, source hashes, review receipts and fresh DEV. |
| `test_history.py` | CPU tests for interruptions, provenance, budgets and decisions. |

Results stay under `runs/actual-history-v3-4gpu/`; individual training arms use sibling folders with that prefix. `result.json` contains the complete comparison, `comparison-1729.json` and optional `comparison-2718.json` preserve each pair, and `budget.json` records charged work. `collection/` retains the collected histories and immutable TRAIN, while evaluation folders retain raw generations and full semantic observations. The terminal prints a short summary and the report path. This selects diagnostic evidence, not a production-ready model.

## Set up once

Use Linux x86-64, Python 3.13, Node.js 22 or later, Git LFS, and a V100 allocation.

```sh
git clone --branch actual-history-v3 https://github.com/wassilysavin/for-mu-cluster.git
cd for-mu-cluster
./cluster python -m venv .venv
source .venv/bin/activate
./cluster git lfs install --local
./cluster git lfs pull
./cluster python -m pip install -r requirements.txt
./cluster python run.py check
```

The repository attributes use Git LFS for weights and TRAIN. Use and distribution of the Gemma weights and derivatives are subject to the restrictions in section 3.2 of the bundled [Gemma Terms of Use](GEMMA-TERMS.md). The model changes are listed in [NOTICE](NOTICE).

## Historical full TRAIN v2

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

## Historical TRAIN-volume comparison

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
