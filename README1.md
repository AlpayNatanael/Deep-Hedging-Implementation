# A basic deep-hedging teaching notebook

We study how a neural network hedges one short call in a synthetic Heston market. The notebook explains hedging, shows two original figures from [*Deep Hedging*](https://arxiv.org/abs/1802.03042), then presents our saved results and matched implementation comparisons. This is a one-call example; portfolio compression is future work.

## Read the notebook

Open `hedging_basics.ipynb`. Its saved figures are embedded, so reading the outputs requires no training. To rerun the cells, open it from this project folder:

```bash
python -m pip install -r environment/requirements.txt
python -m notebook hedging_basics.ipynb
```

**Run All loads saved results and draws seven figures. It does not train or regenerate market paths.** The plotting functions work with either experiment, such as `plots.plot_learning(ours)` or `plots.plot_learning(paper)`, and accept lists for matched comparisons.

## One class for the experiment

```python
from scripts.general_nn import generalNN
from scripts.settings import Settings

settings = Settings(warmup_updates=100, continuation_updates=100, number_of_replicates=1)
model = generalNN(method="ours", settings=settings, dynamic_data=True)
model.train()
validation_scores = model.validate()
model.evaluate()
model.save("results/my_trial")
```

This optional example uses a small practice budget. Choose `method="paper"` for the paper-style network; `Settings(trade_variance=False)` gives its stock-only control. Set `dynamic_data=False` to sample a fixed training pool. The hedge can still trade every day.

`load_data(data, split="validation")` accepts supplied data. `evaluate_historical(data)` returns a separate evaluation view for plotting; it preserves the saved synthetic results. Historical inputs must be supplied, and no historical performance is claimed here.

Start with `scripts/general_nn.py`, `scripts/settings.py`, and `scripts/plots.py`. The [code guide](docs/code_guide.md) explains data shapes, validation, historical evaluation, and saving/loading. Original paper figures remain in `assets/paper/`; saved numerical results and weights remain in `results/`.
