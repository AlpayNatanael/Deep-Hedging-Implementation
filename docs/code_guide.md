# Using the current Python interface

The current root notebooks construct and train new experiments. This guide covers configuration, plotting and persistence. See [Methods](methods.md) for mathematics, [Experiments](experiments.md) for budgets/reproduction and [Historical data](historical_data.md) for supplied observations.

## A small complete training example

Run from the repository root with dependencies installed. These small update/data budgets check the workflow; they are not performance evidence. They keep the default 30-day grid.

```python
from scripts.general_nn import generalNN
from scripts.settings import Settings
from scripts import plots

settings = Settings(
    warmup_updates=2,
    continuation_updates=3,
    number_of_replicates=1,
    batch_size=64,
    normalization_size=128,
    validation_size=128,
    test_size=256,
    validation_interval=1,
    static_training_size=512,
)

model = generalNN(
    method="ours", settings=settings,
    dynamic_data=True, name="Fresh-batch example",
)
model.train(verbose=False)
validation_scores = model.validate()
model.evaluate()
plots.plot_results_grid(model)
```

`train()` fits a shared CVaR warm-up and two final policies. `validate()` measures frozen policies on validation data and records the measurement. `evaluate()` computes test outcomes and returns the experiment. Training does not evaluate the test set. Construct a new object for another fit; a trained object cannot be trained again.

## Independent configuration choices

| Argument / setting | Meaning |
|---|---|
| `method="ours"` / `"paper"` | Preset network/instruments; explicit settings override defaults. |
| `dynamic_data=True` / `False` | New training batches / sample a stored pool with replacement. |
| `data_generator="Regular"` / `"Fast"` | Sequential / parallel built-in generation. Omission and `None` select Regular. |
| `fast_workers=6` | Upper bound on simulation threads, capped at twice the stock count. |
| `fast_prefetch=2` | Fresh/Fast waiting-batch capacity; zero disables prefetch. |
| `trade_variance=False` with the paper preset | Force variance holdings to zero, retaining its channel and variance observations. |
| `retain_training_batches=True` | Retain completed raw batches, increasing memory. Default is False. |

Default widths are 3 → 16 → 16 → 1 with Tanh and 4 → 17 → 17 → 2 with ReLU/normalization. Active feature scaling is shared by both method names. Weights are separate for each date. `dynamic_data` is independent of `method` and defaults to True for both presets.

Using the settings above, these construct two untrained configurations:

```python
fixed = generalNN("paper", settings=settings, dynamic_data=False)
fast = generalNN(
    "paper", settings=settings, dynamic_data=True,
    data_generator="Fast", fast_workers=6, fast_prefetch=2,
)
```

Fast requires Numba. Both built-in backends use Regular for validation/test paths. Construction does not train or start the optional pool. `train()` closes Fast resources even after errors; call `close()` after directly generating paths through lower-level methods.

`NetworkSettings`, `StockSettings` and `InstrumentSettings` configure stocks, layers and channels. To give the compact preset the paper experiment's instruments:

```python
from dataclasses import replace

both = generalNN(
    "ours",
    settings=replace(settings, instruments=fixed.settings.instruments),
    dynamic_data=True,
)
```

Its architecture remains compact with two outputs. The three-stock notebooks instead explicitly match networks and instruments. Use fresh unresolved settings when switching presets: an already resolved `model.settings` has a family label that must agree with `method`.

## Supplied datasets and custom generators

`DataSet` and dictionaries of its fields are accepted. With $B$ scenarios, $N$ intervals, $M$ stocks and $D$ channels:

| Field | Shape | Meaning |
|---|---|---|
| `prices` | `(B, N+1, D)` | Raw per-unit instrument quotes |
| `stock_prices` | `(B, N+1, M)` | Underlying prices used as features |
| `variances` | `(B, N+1, M)` | Variance features, not volatility |
| `payoffs` | `(B,)` | Terminal liability payments |
| `cashflows` | optional `(B, N, D)` | Per-unit interval payments; defaults to zeros |

Numeric arrays become detached CPU float64 tensors. Values must be finite, stock prices positive, and quotes/variances nonnegative. Quotes include all configured channels, including forced-zero holdings. Supplied names, when present, must match settings order. Historical data have additional mandatory fields.

For independently prepared dictionaries:

```python
# Template: use a trained model and supply independent validation_data/test_data.
model.load_data(validation_data, split="validation")
validation_scores = model.validate()
model.evaluate(data=test_data)
```

For supplied training data, use `dynamic_data=False` and `training_data=...`, or `load_data(..., split="training")` before fitting. Shape, reserved-seed, identical-whole-dataset and available date-overlap checks reject known reuse; they cannot prove independence or detect every partial overlap.

A custom generator is a callable `(size, numpy_generator, resolved_settings)` returning a `DataSet`/dictionary with exactly `size` scenarios. It handles every requested role, including held-out generation, unlike built-in Fast's Regular evaluation path. Other joint dynamics and observed instruments need suitable supplied data/custom generation. Callbacks are not serialized.

## Plot fitted experiments and reports

```python
plots.plot_results_grid(model, objective="CVaR", explain=True)
plots.plot_learning(model, objective="Entropic")
plots.plot_risk(model)
plots.plot_loss_tails(model, objective="CVaR")
```

The default grid has learning, test risk and tails in three rows with CVaR/Entropic columns. `objective="CVaR"` selects one column. Ordinary plots accept one experiment or a matched list and case-insensitive objective names. Incompatible settings/test observations are rejected. Helpers supporting `show=False` return their documented figure/axes objects; the grid returns `(figure, axes)`. Most support PNG/SVG export through `save_dir`.

| Other public helper | Purpose |
|---|---|
| `plot_learning_simple` | Backward-compatible single-objective/run learning plot |
| `plot_losses` | Full-range terminal-loss distributions |
| `plot_pnl_histogram` | Profit distributions with a common premium and optional stored stock-delta reference |
| `plot_paths` | Example first-stock paths and corresponding liability payments |
| `show_published_figure` | Display stored source-paper artwork, rather than local reproduction results |

For one synthetic call, `model.evaluate(include_heston=True)` adds its Heston reference. `plots.plot_pnl_comparison([reference, other], premium=common_premium, include_heston=True)` needs compatible evaluated models and an already evaluated Heston reference. This helper displays directly and has its own signature rather than the ordinary `show`/`save_dir` controls. This hedge is not supported for multi-stock or historical liabilities.

Report plotting uses small report dictionaries rather than fitted model objects:

| Helper | Input |
|---|---|
| `plots.plot_memory_comparison(reports, save_dir=...)` | Two `run_memory_experiment` reports |
| `plots.plot_path_generation_comparison(generation_reports, save_dir=...)` | `run_path_benchmark` reports |
| `plots.plot_path_training_comparison(training_reports, save_dir=...)` | Four-condition isolated training reports |
| `plots.show_path_speedups(generation_reports, training_reports)` | Both lists containing all four conditions |
| `plots.show_table(headers, rows)` | Table data; `show=False` returns Markdown |

These helpers do not train or generate data. The path-training helper with `show=False` returns a dictionary of training and risk figure/axes pairs.

`run_memory_experiment` and `run_path_benchmark` require fresh built-in synthetic configurations, not models with runs or loaded data. They use child processes through the active interpreter and leave parents untrained. See [Experiments](experiments.md) for scopes/output paths.

## Inspect and preserve a new run

```python
model.save("results/my_run")
restored = generalNN.load("results/my_run")
plots.plot_risk(restored)
```

The directory contains `experiment.pt`, preserving settings, weights, stored data, histories, results and historical views. Fixed-pool snapshots can be large. Optimizer state is absent, so snapshots do not resume interrupted fits. Generator options are retained; old supported snapshots default to Regular when absent. Reconnect custom callbacks before generating additional data; prediction on supplied data does not need them.

Inspect `settings`, `parameter_count`, `network`, `models`, `data`, `runs`, `arrays`, `metadata`, `batch_records`, `training_batches` and `measurements`. `get_run`, `losses`, `costs` and `predict` select an objective/replicate. `network` exposes the first selected policy; `models` retains all selected policies.

`generalNN.from_saved(family)` accepts `"ours"`, `"paper_full"` or `"paper_stock"` and reads older `results/summary.json`, arrays and checkpoints without fitting. It differs from loading a new `experiment.pt`. These archives have a [known replay consistency issue](experiments.md#software-checks-and-known-issues); old reported losses should not be assumed to reproduce under the current forward path.
