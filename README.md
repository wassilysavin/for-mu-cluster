# GEM-pRF Functions

Train and evaluate FunctionGemma on GEM-pRF parameter questions. Each run records the architecture, dataset and checkpoint identities together with its evaluation. The training code and evaluators run in Python.

## Start here

From the repository root, with the [project environment](docs/experiments.md#environment) active and Git LFS assets available, verify the weekly TRAIN on CPU.

```sh
./cluster python -u extended_train.py check
```

The published dataset has **16,174 examples**. Its size comparison uses a matched subset of **5,169**. The command verifies every input and target with the frozen tokenizer. See the [dataset audit](docs/train-audit.md) for coverage and limitations.

| Task | Command |
|---|---|
| Inspect weekly dataset coverage | `./cluster python audit_week_train.py` |
| Check the historical dialogue experiment inputs | `./cluster python run.py --suite history check` |
| Inspect the saved paraphrase/contrast comparison | `./cluster python inspect_transfer.py` |
| Measure TRAIN fit of both saved checkpoints on four allocated GPUs | `./cluster python -u check_new_train_fit.py --gpus 4` |
| Compare 5,169 and 16,174 TRAIN examples after TRAIN fit completes | `./cluster python -u week_compare.py --gpus 4` |
| Read the saved weekly comparison across all DEV panels | `./cluster python -m diagnostics.inspect_week_comparison` |
| Inspect saved DEV155 losses and new DEV1250 critical flags | `./cluster python -m diagnostics.inspect_week_failures` |
| Trace the shared critical cases and validate the arguments behind DEV155 losses | `./cluster python -m diagnostics.inspect_week_failure_details` |
| Verify the targeted TRAIN supplement and fresh dialogue DEV on CPU | `./cluster python -m training.targeted_dialogues check` |
| Run the regression tests on CPU | `./cluster python -m unittest discover -s tests -t .` |

The GPU diagnostic uses the completed semantic-transfer run and its two step-8151 checkpoints. The weekly dataset check runs independently. Installation, experiment-specific dependencies, training commands, original limits and restart procedures are in the [experiment guide](docs/experiments.md).

## Repository layout

| Location | Contents |
|---|---|
| Root commands and `experiments*.json` | Experiment entry points and frozen training recipes |
| [model.py](model.py), [ledger.py](ledger.py) | FunctionGemma implementation, checkpoint handling and run identities |
| [training/](training/) | Paired training preparation and checkpoint comparison policies |
| [data_tools/](data_tools/) | Dataset construction, replay, prompt authoring and split checks |
| [evaluation/](evaluation/) | Function-call contracts, dialogue state and grading rules |
| [eval_tools/](eval_tools/) | Commands for streaming and scoring TRAIN and DEV observations |
| [diagnostics/](diagnostics/) | Readers and reports for saved runs, plus TRAIN-fit workers |
| [tests/](tests/) | Regression and equivalence tests |
| [assets/](assets/) | Tracked weights, tokenizer, datasets and contracts |
| `runs/`, `.runtime/` | Local results and private scratch space |

The launcher keeps caches and temporary files inside the checkout. Model jobs share an execution lease. Historical source receipts identify the implementation that produced each saved result. A partial run requires its matching implementation to resume. Redistribution of Gemma assets is subject to the [Gemma terms](GEMMA-TERMS.md) and [notice](NOTICE).
