# Running and reproducing the experiments

Run from the repository root. The current entry points are the four notebooks below. Older files in `misc/` and saved archives may use different settings.

## Environment and execution

For example, using Python 3.12 on macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name deep_hedging --display-name "Python (deep_hedging)"
python -m notebook
```

Select that kernel in Jupyter. On Windows, adapt the activation command. Requirements specify ranges; [tested_versions.txt](../environment/tested_versions.txt) records an earlier saved run, not an exact lock for every artifact. Record package versions and hardware for each new timing comparison.

Fast needs Numba in the kernel's environment. If necessary run `%pip install "numba>=0.61,<1"` in a cell, restart the kernel and rerun all measurements. Regular can fall back to Python without Numba, producing a different performance baseline. Check:

```python
import sys
import numba
from scripts.general_nn import SIMULATION_BACKEND

print(sys.executable)
print(numba.__version__)
print(SIMULATION_BACKEND)  # Expected: numba
```

Experiments use CPU float64 and one PyTorch thread per training process. Fast adds simulation threads. Restart and Run All after changing settings. `train()` rejects an already-trained object; construct a new experiment for another fit.

## Notebook map and configured budgets

| Notebook | Question | Configuration |
|---|---|---|
| [Single_Call_Comparison](../Single_Call_Comparison.ipynb) | Effects of network/data choices and tradable instruments | Four configurations, first with costs and then separately with `cost_rate=0`; Heston P&L comparisons use the zero-cost run. |
| [Small_Portfolio_Comparison](../Small_Portfolio_Comparison.ipynb) | Stored versus fresh data with the same network | Three independent stocks/calls, six instruments, matched 36/36 Tanh networks with batch normalization. |
| [Large_Portfolio_Memory_Comparison](../Large_Portfolio_Memory_Comparison.ipynb) | RAM/runtime consequences of data mode | The same three-stock setup, measured in isolated child processes. |
| [Path_Generation_Comparison](../Path_Generation_Comparison.ipynb) | What parallel generation adds to each data mode | Fixed/Regular, Fixed/Fast, Fresh/Regular and Fresh/Fast; identical explicit networks and the paper preset in every case. |

The memory notebook's name describes its large stored scenario pool; it still has three stocks. Older notebook text referring to `Large_Portfoloio_Comparison.ipynb` describes the experiment now named `Small_Portfolio_Comparison.ipynb`.

| Setting | Single call, each cost regime | Small portfolio / memory | Path: quick | Path: full |
|---|---:|---:|---:|---:|
| Stocks | 1 | 3 | 3 | 3 |
| Warm-up updates | 100 | 100 | 10 | 100 |
| Continuation updates per objective | 100 | 1,000 | 25 | 1,000 |
| Batch size | 256 | 256 | 256 | 256 |
| Fixed pool size | 25,600 | 262,144 | 8,192 | 262,144 |
| Calibration sample per stage | 2,048 | 2,048 | 512 | 2,048 |
| Validation scenarios | 2,048 | 2,048 | 1,024 | 2,048 |
| Test scenarios | 8,192 | 8,192 | 2,048 | 8,192 |
| Validation interval | 25 | 50 | 5 | 50 |
| Replicates inside each model | 1 | 1 | 1 | 1 |
| Notebook-level repeats | 1 | 1 | 3 | 3 |

All use 30 daily intervals, 64 substeps per interval, CVaR level 0.5 and entropic risk aversion 1. Costs are 0.001 except the zero-cost single-call block. Three-stock parameters differ between stocks; inspect `stocks_list` when comparing modified experiments.

The single-call comparison changes architecture and data mode together. The three-stock comparison matches networks and instruments. All neural configurations observe variance, including stock-only cases. The built-in Heston hedge supports one call only; the small-portfolio notebook currently plots neural policies without invoking it.

The single-call zero-cost block creates new objects but reuses names such as `paper` and `ours`. After Run All those names refer to zero-cost models. Associate printed scores/figures with their own block; explanatory text can retain older values.

## Seeds and fair comparisons

Default initialization/training/validation/test seeds are 7,000 / 10,100 / 20,200 / 40,400. Internal replicate $r$ adds $r$ to initialization and $100r$ to training seeds. Warm-up uses that training seed; both continuations use seed plus one, so objective branches intentionally can reuse scenario draws.

The path notebook's outer repeat $r$ adds $1000r$ to initialization/training seeds and holds evaluation seeds fixed. Configuration order is randomized. Four configurations × three repeats gives 12 child runs per part, 24 overall. Fast/Regular training arrays may differ; evaluation scenarios match by content identifiers.

Fresh mode creates new batches within a stage. Training comparisons equalize updates, not total generated scenarios or distinct draws across objectives. Three repeats measure training variability, not population-risk confidence intervals.

## Memory and path measurements

`run_memory_experiment` trains/evaluates a new configuration copy in a child process; parent weights/data are not transferred or trained. It samples child RSS every 0.01 seconds:

- `baseline_mib`: RAM immediately before measurement.
- `peak_mib`: largest sampled total RAM, including libraries and scratch arrays.
- `peak_extra_mib`: peak minus baseline, not total RAM.
- `retained_dataset_mib`: deduplicated tensor storage remaining after evaluation, distinct from peak RSS.

Elapsed time includes preparation, training and evaluation, excluding startup and simulator warm-up. Notebook/GPU memory is excluded. Brief spikes may be missed.

The path-only test creates exactly the pool count in every case: one retained dataset or batches of 256 consumed and discarded. Quick mode consumes 32 batches, full mode 1,024. Quotes, payoffs, zero cashflows, dataset checks and inspection are included. It separately reports warm-up, CPU seconds and largest consumer dataset. Queued/working batches contribute to RSS even though the consumer-size column counts one batch.

For controlled timing, avoid competing training jobs and record CPU/worker/thread settings. Compare Regular/Fast within a data mode, then Fixed/Fast with Fresh/Fast. More CPU capacity can reduce elapsed time without reducing total CPU work.

## Outputs and interpreting results

The single-call and small-portfolio notebooks display results but do not automatically save new snapshots. Use `model.save(...)` to preserve a run. Files under `results/single_call_option/` are older artifacts and are not refreshed by Run All.

The memory notebook writes reports/logs under `results/portfolio_memory_comparison/`. The path notebook writes `results/path_generation_comparison/quick/` or `full/`, containing combined `generation_reports.json` / `training_reports.json`, per-repeat JSON/logs and PNG/SVG figures. Reruns overwrite corresponding files; copy the directory to preserve a run.

Path plots show median timing/RAM and mean risk, with sample standard deviation across training seeds in the risk table. Each score uses its objective-specific policy. Lower loss risk is better; P&L uses a common premium. A positive histogram peak alone does not establish a better hedge.

Committed memory reports give roughly 2,522.9 versus 551.5 MiB peak RAM and CVaR 8.699 versus 8.731 for fixed/fresh modes: one seed, 78.1% less peak RAM and 70.5% longer elapsed time. This is not statistical equivalence evidence. Committed path outputs use quick budgets; full-training runtime parity and portfolio-information compression remain open.

## Software checks and known issues

After installing dependencies:

```bash
python -m unittest discover -s tests -v
python checks/check_results_grid.py
python checks/check_memory_tracking.py
```

These check selection, kernel agreement, held-out paths, prefetch cleanup, persistence, plotting and isolated memory measurement on small fixtures. They do not establish financial performance or convergence.

The broader optional check is:

```bash
python checks/check_refactor.py
```

It requires legacy `results/summary.json`, `arrays.npz` and checkpoint folders. Legacy example-path replay has failed with `Legacy example-path accounting changed.` on both unchanged and generator-updated source in the tested environment. All 18 old checkpoints produced identical predictions across those two versions; disagreement with stored arrays remains unresolved. Older JSON check reports describe their original runs, not guarantees that every current check passes. Prefer freshly trained snapshots for new reproducibility claims.

When troubleshooting, verify the kernel/interpreter for missing Numba, restart after import changes, and keep comparison settings/test identifiers matched. Supplied observations must follow [the historical-data contract](historical_data.md).
