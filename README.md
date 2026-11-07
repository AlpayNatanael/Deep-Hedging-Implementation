# Deep Hedging: Implementation and Memory-Efficient Training

**Can we preserve hedging quality while reducing the memory needed to train a neural hedging strategy?**

We implement and compare paper-inspired and compact neural hedging networks in a simulated Heston market. We study stock-only and stock-plus-variance trading, transaction costs, and training under CVaR and entropic risk. The project builds on [*Deep Hedging* by Buehler et al.](https://arxiv.org/abs/1802.03042).

Our main modification is to generate fresh market scenarios during training instead of retaining a large fixed training dataset.

**In a matched three-stock experiment, fresh-batch training achieved similar observed test CVaR using approximately 22% of the peak CPU RAM required by fixed-pool training.** That run took longer to complete. An optional parallel path generator is now implemented to investigate this runtime trade-off.

## What we implement

- **Daily neural hedging policies.** At each trading date, a network observes current stock prices, variances and previous holdings, then chooses positions in the available instruments. The default designs use two hidden layers: a compact Tanh network and a paper-inspired ReLU network with batch normalization.
- **Stock and variance-claim trading.** Experiments hedge one call or a portfolio of calls, using stocks alone or stocks together with claims linked to accumulated variance. Stock-only neural configurations still observe variance; the restriction concerns what they can trade.
- **Risk-based training and calibration.** A shared CVaR warm-up is followed by separate CVaR and entropic continuations. In the configured examples, CVaR averages the worst half of losses; entropic risk penalizes large losses exponentially. Where batch normalization is used, its statistics are calibrated from training-source samples. Validation selects checkpoints; separate test scenarios measure final performance.
- **Fixed-pool and fresh-batch training.** Network design and training-data mode are configured independently. By default, fresh mode discards completed training batches while retaining the evaluation and calibration data needed by the experiment.
- **Model-based Heston benchmarks.** The single-call notebook includes a separate zero-transaction-cost experiment comparing neural strategies with Heston hedges under matched trading instruments and dates.
- **Memory tracking and parallel simulation.** Isolated CPU processes measure RAM and runtime. The optional Fast backend generates independent path blocks concurrently and prepares upcoming training batches through a bounded queue.

## Experiments

| Notebook | What it tests |
|---|---|
| [Single_Call_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison.ipynb) | One call: four neural configurations, stock-only or stock-plus-variance trading, and a separate cost-free Heston comparison. Network architecture and data mode change together in the neural comparison. |
| [Large_Portfoloio_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfoloio_Comparison.ipynb) | Three calls on three independent simulated stocks. Match network architectures and instruments to isolate fixed-pool versus fresh-batch training. |
| [Large_Portfolio_Memory_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfolio_Memory_Comparison.ipynb) | Run the matched three-stock configurations in separate processes and compare peak RAM, runtime and test risk. |
| [Path_Generation_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Path_Generation_Comparison.ipynb) | Compare Regular/Fast generation with fixed pools/fresh batches. Measure equal-scenario creation first, then complete preparation, training and evaluation. |

## Preliminary memory result

The committed three-stock memory experiment uses a fixed pool of 262,144 scenarios versus fresh training batches of 256 scenarios. Both policies have the same architecture, 64,980 parameters and held-out test scenarios. Both trade the three stocks and their three variance claims. The proportional transaction-cost rate is 0.001, or 0.1% of traded notional.

| Measurement | Fixed pool | Fresh batches |
|---|---:|---:|
| Sampled peak CPU RAM | 2,522.9 MiB | 551.5 MiB |
| Test CVaR | 8.699 | 8.731 |
| Test entropic risk | 18.553 | 17.328 |
| Preparation + training + evaluation | 216.4 s | 368.8 s |

This run used **78.1% less sampled peak RAM** with fresh batches and took **70.5% longer**. Each risk score evaluates the policy trained for that objective; lower is better. Risk is measured on hedging losses before receiving the option premium. P&L comparisons add the same premium to every strategy.

These are observations from one training seed, rather than an established equivalence result. The memory improvement concerns retained training data; the matched policies have the same parameter count.

RAM is sampled approximately every 10 milliseconds and includes the training process's libraries, models, datasets and temporary arrays. Timing includes data preparation, training and test evaluation after simulator warm-up. The parent notebook and GPU memory are excluded. Brief memory spikes may be missed, and elapsed times depend on the machine.

See the [saved reports and memory plot](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/tree/main/results/portfolio_memory_comparison).

## Fast path generation

The optional Fast backend is integrated into `generalNN` for both fixed-pool and fresh-batch training. It preserves the existing float64 simulation scheme while distributing independent path blocks across CPU workers. Fresh training can also use bounded batch prefetch.

The path-generation notebook compares all four combinations using the same network and instruments. Its creation benchmark uses equal scenario counts; its training benchmark uses equal optimizer-update budgets. It reports elapsed time, CPU time, memory and held-out risk. Validation and test scenarios are shared across the built-in backends; training draws can differ because random streams are partitioned differently.

The notebook starts with `QUICK_RUN = True` and three training seeds. Set it to `False` for the full experiment. The committed quick results verify operation and provide preliminary measurements; full-scale training runtime parity remains unestablished. Parallel speedups use additional CPU capacity.

## Getting started

From the repository folder, install the dependencies and start Jupyter using the same Python environment:

```bash
python -m pip install -r environment/requirements.txt
python -m notebook
```

Open a comparison notebook and run its cells in order. Saved outputs can also be inspected on GitHub without retraining. Run All in the memory notebook trains the two configurations sequentially in separate processes.

**Regular generation is the default.** Existing model constructors can omit `data_generator`. Select `data_generator="Fast"` in a model constructor to opt in. Fast generation requires Numba, which is included in the requirements file.

If you are already running a notebook and Numba is missing, install it into that notebook's environment:

```python
%pip install "numba>=0.61,<1"
```

Restart the kernel and rerun all cells so both Regular and Fast measurements use the intended compiled environment.

## Code and documentation

- `scripts/general_nn.py`: datasets, daily networks, simulation, training, calibration, evaluation and persistence.
- `scripts/settings.py`: market parameters, instruments, network designs and experiment settings.
- `scripts/plots.py`: learning curves, risk comparisons, loss tails, P&L figures and comparison grids.
- `scripts/memory_tracker.py`: sampled CPU RAM measurement and isolated training experiments.
- `scripts/fast_paths.py`: optional parallel generation and bounded prefetch.
- `scripts/path_benchmark.py`: isolated, equal-scenario creation benchmarks.

See the [code guide](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/docs/code_guide.md) for model configuration and persistence, and the [historical-data guide](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/docs/historical_data.md) for evaluating supplied observations. The experiments reported here use simulated data.

## Research direction

The current experiments study training-data efficiency and simulation speed. Our longer-term goal is portfolio compression: representing an option portfolio with less information while preserving hedging performance. Further experiments will assess stability across training seeds, larger portfolios and full-budget accelerated training.
