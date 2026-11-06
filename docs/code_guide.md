# Using `generalNN`

`generalNN` owns the settings, network, data, training, validation, and results. The notebook keeps the same seven figures; its default cells only load the saved synthetic benchmark.

## Load the existing results

```python
from scripts.general_nn import generalNN
from scripts.settings import Settings
from scripts import plots

ours = generalNN.from_saved("ours")
paper = generalNN.from_saved("paper_full")
control = generalNN.from_saved("paper_stock")

plots.plot_learning(ours)
plots.plot_learning(paper)
plots.plot_risk([paper, control, ours])
```

The saved names remain supported. The plotting functions accept one object or a matched list; comparisons check the test data and relevant market, cost, and risk settings.

## Configure and train

```python
settings = Settings(
    warmup_updates=100,
    continuation_updates=100,
    number_of_replicates=1,
    normalization_size=2048,
    validation_size=2048,
    test_size=8192,
    static_training_size=8192,
)

model = generalNN(method="ours", settings=settings, dynamic_data=True)
model.train()
validation_scores = model.validate()
model.evaluate()
plots.plot_learning(model)
model.save("results/my_run")
```

This short run demonstrates the API. The saved benchmark used a larger budget. Training keeps the CVaR warm-up and the independent CVaR/entropic continuations; validation selects checkpoints, and test evaluation measures their held-out performance.

The required `method` is `"ours"` or `"paper"`. Defaults preserve our $3\to16\to16\to1$ Tanh network and the paper-style $4\to17\to17\to2$ network with batch normalization and ReLU. To use the paper's network with its variance holding fixed at zero:

```python
stock_control = generalNN(method="paper", settings=Settings(trade_variance=False))
```

`NetworkSettings`, `StockSettings`, and `InstrumentSettings` are available from `scripts.settings` for explicit extensions. The built-in multi-stock Heston generator uses independent stocks. Other joint models and arbitrary observed instruments need supplied data or a custom `data_generator`.

For example, our design can also trade two instruments:

```python
from scripts.settings import InstrumentSettings

settings = Settings(instruments=(
    InstrumentSettings("Stock"),
    InstrumentSettings("Stock variance", kind="variance", multiplier=1000.0),
))
two_instruments = generalNN(method="ours", settings=settings)
```

This explicit configuration uses four inputs and two outputs while keeping the two 16-neuron Tanh hidden layers. The default one-instrument configuration and its saved weights are unchanged. Inputs contain the price and variance of each modeled stock followed by the previous holding in each configured instrument; outputs follow the instrument order.

`dynamic_data=True` generates fresh training paths at each update. `dynamic_data=False` samples a fixed training pool with replacement. This flag changes the training data, not whether holdings change over time. Validation and test data remain separate in either mode. No performance advantage of one data mode is claimed by the saved benchmark.

## Load data and validate

`load_data(data, split=...)` accepts a data dictionary. With $B$ paths, $T$ trading intervals, $D$ traded instruments, and $M$ underlying stocks, provide:

| Key | Array shape | Meaning |
|---|---|---|
| `prices` | `(B, T+1, D)` | Prices of the configured trading instruments |
| `stock_prices` | `(B, T+1, M)` | Underlying stock prices used as features |
| `variances` | `(B, T+1, M)` | Corresponding variance features |
| `payoffs` | `(B,)` | Liability payment on each path |

Dates include the initial and final observations. Instrument ordering and units must match the settings. In the default paper experiment, the variance claim's raw price is converted into contract lots by the configured notional; do not apply that conversion twice.

For supplied arrays already collected into dictionaries:

```python
model.load_data(validation_data, split="validation")
validation_scores = model.validate()

model.load_data(test_data, split="test")
model.evaluate()
```

For a fixed supplied training set, construct with `dynamic_data=False` and pass `training_data=training_data`, or load it with `split="training"` before calling `train()`. Keep training, validation, and test observations separate. Loading checks the data structure; `validate()` measures the fitted policies on validation data and stores the metrics.

## Evaluate supplied historical observations

```python
historical = model.evaluate_historical(observed_data, name="Historical evaluation")
plots.plot_losses(historical)
plots.plot_risk(historical)
```

`observed_data` uses the same four arrays. Supply chronological held-out windows, their realized liability payoffs, and variance features available at each decision date. Every traded instrument needs appropriate observed prices; missing market quotes are not fabricated. The returned object supports the same plotting interface and leaves the original synthetic results unchanged. Historical data are not included in this project, so there are no historical performance results to report.

The complete required names, timestamps, provenance and cashflow conventions are in [Historical data](historical_data.md). Historical plots use the supplied payments rather than constructing an option payoff from stock prices.

`plot_learning(historical)` still shows that policy's original training history. The loss, risk and tail plots show its new historical evaluation. Calling `historical.evaluate()` rechecks the stored observations without generating synthetic replacements.

## Inspect and save

The main attributes remain `settings`, `network`, `models`, `data`, `runs`, `arrays`, `batch_records`, and `training_batches`. They expose the configuration, selected models, retained datasets, training histories, scores, and individual test outcomes.

Raw training batches are discarded by default. Set `retain_training_batches=True` to retain them, using additional memory. The original benchmark's raw streaming batches were not retained.

```python
model.save("results/my_run")
restored = generalNN.load("results/my_run")
plots.plot_learning(restored)
```

Saving retains fitted weights, data and measured results, including historical views. It does not resume an interrupted optimizer. Custom generator functions are not serialized; reconnect `restored.data_generator` before generating additional data with such an experiment.

The implementation is centered on `scripts/general_nn.py`, with configuration in `scripts/settings.py` and plotting in `scripts/plots.py`. The default notebook's numerical figures remain unchanged.

The optional verification script is `checks/check_refactor.py`; its completed report is `results/refactor_check.json`.
