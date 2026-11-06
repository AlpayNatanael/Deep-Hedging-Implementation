"""One experiment class for both neural designs, data modes and evaluations.

The default networks and the supplied benchmark are preserved. Numerical
Heston and risk helpers live at the end of this file to keep imports small.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts.settings import FAMILIES, OBJECTIVES, Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataSet: # Stores market scenarios, instrument prices, and the payments we owe.
    """Prices are raw per-unit quotes; payoffs are terminal currency payments."""
    # Raw price per unit of each trading instrument: (B, N + 1, D).
    prices: torch.Tensor

    # Underlying stock prices used as network inputs: (B, N + 1, M).
    stock_prices: torch.Tensor

    # Variance of each stock used as a network input: (B, N + 1, M).
    variances: torch.Tensor

    # Total liability payment owed at maturity in each scenario: (B,).
    payoffs: torch.Tensor

    # Optional payments per instrument unit during each interval: (B, N, D).
    cashflows: torch.Tensor | None = None

    # Recorded random seed associated with generating these data, if known.
    seed: int | None = None

    # Purpose of these data, such as "training", "validation", or "test".
    role: str = "training"

    # Observation timestamps for every scenario: (B, N + 1).
    dates: object = None

    # Latest input-availability timestamp for each trading decision: (B, N).
    feature_times: object = None

    # Instrument names, ordered to match the last dimension of prices.
    instrument_names: object = None

    # Stock names, ordered to match stock_prices and variances.
    stock_names: object = None

    # Extra information, such as data sources and accounting conventions.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("prices", "stock_prices", "variances", "payoffs"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float64).detach().cpu()
            setattr(self, name, value)
        if self.stock_prices.ndim == 2:
            self.stock_prices = self.stock_prices.unsqueeze(-1)
        if self.variances.ndim == 2:
            self.variances = self.variances.unsqueeze(-1)
        if self.prices.ndim != 3 or not len(self.prices):
            raise ValueError("prices must have shape (paths, dates, instruments).")
        if self.stock_prices.ndim != 3 or self.stock_prices.shape[:2] != self.prices.shape[:2]:
            raise ValueError("stock_prices must have shape (paths, dates, stocks).")
        if self.variances.shape != self.stock_prices.shape or self.payoffs.shape != (len(self),):
            raise ValueError("Match the variance states and provide one payoff per path.")
        if self.cashflows is None:
            if self.metadata.get("asset_cashflows") == "included":
                raise ValueError("Supply the asset cashflows declared as included.")
            self.cashflows = torch.zeros_like(self.prices[:, :-1])
        else:
            self.cashflows = torch.as_tensor(self.cashflows, dtype=torch.float64).detach().cpu()
        if self.cashflows.shape != self.prices[:, :-1].shape:
            raise ValueError("cashflows must have shape (paths, trading intervals, instruments).")
        values = (self.prices, self.stock_prices, self.variances, self.payoffs, self.cashflows)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("Every observation and payoff must be finite.")
        if torch.any(self.prices < 0) or torch.any(self.stock_prices <= 0) or torch.any(self.variances < 0):
            raise ValueError("Use nonnegative asset quotes and variance, and positive stock prices.")
        for name in ("dates", "feature_times"):
            value = getattr(self, name)
            if value is not None:
                dates = np.asarray(value, dtype="datetime64[ns]")
                setattr(self, name, np.datetime_as_string(dates, unit="ns").tolist())

    def __len__(self):
        return len(self.prices)

    def select(self, indices, role="training_batch"):
        """ selects entire scenarios, preserving all dates and instruments belonging to those scenarios."""
        values = {name: getattr(self, name)[indices]
                  for name in ("prices", "stock_prices", "variances", "payoffs", "cashflows")}
        for name in ("dates", "feature_times"):
            value = getattr(self, name)
            values[name] = None if value is None else np.asarray(value)[indices].tolist()
        return DataSet(**values, seed=self.seed, role=role,
                       instrument_names=self.instrument_names, stock_names=self.stock_names,
                       metadata=copy.deepcopy(self.metadata))

    def fingerprint(self, scenarios_only=False):
        """creates an identifier based on the contents"""
        names = ("stock_prices", "variances", "payoffs") if scenarios_only else (
            "prices", "stock_prices", "variances", "payoffs", "cashflows")
        digest = hashlib.sha256()
        for name in names:
            value = getattr(self, name).numpy()
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        if self.dates is not None:
            digest.update(np.asarray(self.dates, dtype="datetime64[ns]").tobytes())
        return digest.hexdigest()


class _TradingPolicy(nn.Module): #main thing to read to understand what the NN sees.
    """Independent daily networks, with previous holdings fed to the next day."""
    def __init__(self, settings, method):
        # Initialize the parent PyTorch class so it can track layers and parameters.
        super().__init__()

        # Store the settings that describe the market, instruments, and network.
        self.settings = settings

        # Store the policy version, "ours" or "paper", which controls input scaling.
        self.method = method

        # Create an empty PyTorch list to hold a separate network for each trading day.
        self.networks = nn.ModuleList()

        # Choose Tanh or ReLU as the activation function for the hidden layers.
        activation = nn.Tanh if settings.network.activation == "tanh" else nn.ReLU
            
        for day in range(settings.number_of_days):
            layers = []
            previous_size = settings.input_size
            for width in settings.network.hidden_sizes:
                layers.append(nn.Linear(previous_size, width, dtype=torch.float64))
                if settings.network.batch_normalization:
                    layers.append(nn.BatchNorm1d(width, eps=settings.batchnorm_epsilon,
                                  momentum=settings.batchnorm_momentum, dtype=torch.float64))
                layers.append(activation())
                previous_size = width
            layers.append(nn.Linear(previous_size, settings.output_size, dtype=torch.float64))
            self.networks.append(nn.Sequential(*layers))

    def forward(self, stock_prices, variances):
        """Calculate daily holdings from the supplied market paths."""

        # Start each path with zero holdings in every trading instrument.
        previous = stock_prices.new_zeros(
            len(stock_prices),
            self.settings.output_size,
        )

        # Collect the holdings chosen on each trading day.
        decisions = []

        # Use the corresponding network for each trading day.
        for day, network in enumerate(self.networks):

            # Collect the input features for the current day.
            features = []

            # Add the current price and variance features for each stock.
            for stock_index, stock in enumerate(self.settings.stocks):

                # Read this stock's price for every path on the current day.
                price = stock_prices[:, day, stock_index]

                # Read this stock's variance for every path on the current day.
                variance = variances[:, day, stock_index]

                # Compute a fixed reference scale for log-price movements.
                price_scale = math.sqrt(
                    stock.initial_variance * self.settings.maturity
                )

                # Express the price relative to the strike and reference scale.
                scaled_price = torch.log(price / stock.strike) / price_scale

                # Express variance as a relative change from its initial value.
                scaled_variance = variance / stock.initial_variance - 1

                # Add both features using the same scaling for every batch.
                features.extend([scaled_price, scaled_variance])

            # Add the previous holding in each trading instrument.
            for instrument_index in range(self.settings.output_size):
                features.append(previous[:, instrument_index])

            # Arrange the features into one row per path.
            inputs = torch.stack(features, dim=1)

            # Pass today's inputs through today's network to choose holdings.
            holdings = network(inputs)

            # Force holdings to zero for instruments disabled in the settings.
            if any(
                not instrument.tradable
                for instrument in self.settings.instruments
            ):
                holdings = torch.stack(
                    [
                        holdings[:, instrument_index]
                        if instrument.tradable
                        else torch.zeros_like(holdings[:, instrument_index])
                        for instrument_index, instrument
                        in enumerate(self.settings.instruments)
                    ],
                    dim=1,
                )

            # Save the holdings chosen for this day.
            decisions.append(holdings)

            # Use today's holdings as the previous holdings on the next day.
            previous = holdings

        # Return holdings with shape: (paths, trading days, instruments).
        return torch.stack(decisions, dim=1)
            
            
    def forward1(self, stock_prices, variances):
        previous = stock_prices.new_zeros(len(stock_prices), self.settings.output_size)
        decisions = []
        for day, network in enumerate(self.networks):
            features = []
            for index, stock in enumerate(self.settings.stocks):
                price = stock_prices[:, day, index]
                variance = variances[:, day, index]
                
                # Use scaled price and variance inputs for our policy.
                if self.method == "ours":

                    # Compute a reference scale for log-price movements
                    # using the initial variance and the total time horizon.
                    scale = math.sqrt(
                        stock.initial_variance * self.settings.maturity
                    )

                    features.extend([
                        # Measure the price relative to the strike,
                        # then divide by the reference scale.
                        torch.log(price / stock.strike) / scale,

                        # Measure how much the current variance differs
                        # from its initial value, as a proportion of that value.
                        variance / stock.initial_variance - 1,
                    ])

                else:
                    # Use the paper-style inputs without the extra scaling.
                    features.extend([
                        # Use the logarithm of the current stock price.
                        torch.log(price),

                        # Use the current variance in its original units.
                        variance,
                    ])
                    
            features.extend(previous[:, index] for index in range(self.settings.output_size))
            holdings = network(torch.stack(features, dim=1))
            if any(not item.tradable for item in self.settings.instruments):
                holdings = torch.stack([
                    holdings[:, index] if item.tradable else torch.zeros_like(holdings[:, index])
                    for index, item in enumerate(self.settings.instruments)
                ], dim=1)
            decisions.append(holdings)
            previous = holdings
        return torch.stack(decisions, dim=1)


class generalNN:
    """
    Create neural network, data, training history, and results together.

    Chose which network design to use "ours" or "paper"
        
    Choose where the training data comes from (technically redundent with the above)
        dynamic_data=True: generate new simulated scenarios for each update.
        dynamic_data=False: randomly select scenarios from a fixed dataset.
    """
    def __init__(self, method, settings=None, dynamic_data=True, training_data=None,
                 data_generator=None, name=None):
        if not isinstance(dynamic_data, bool):
            raise TypeError("dynamic_data must be a boolean.")
        # Store the network preset: "ours" or "paper".
        # This does not choose between static and dynamic training data.
        self.method = method

        # Use the supplied settings, or create default settings if none were given.
        # Then fill in the network and instrument choices for the selected preset.
        self.settings = (settings or Settings()).resolve(method)

        # Store the experiment family: "ours", "paper_full", or "paper_stock".
        self.family = self.settings.method

        # Use the supplied display name, or the default name for this family.
        self.name = name or FAMILIES[self.family]

        # If no name was supplied and the stock or instrument counts differ
        # from the usual preset, create a name showing the instrument count.
        if name is None and (
            len(self.settings.stocks) != 1
            or self.settings.output_size != (1 if method == "ours" else 2)
        ):
            self.name = f"{method}: {self.settings.output_size} instruments"

        # True: generate fresh training batches.
        # False: sample batches from a fixed training dataset.
        self.dynamic_data = dynamic_data

        # Store an optional custom function for generating market data.
        # If this is None, use the built-in market simulator.
        self.data_generator = data_generator

        # Build the initial trading policy, with one network per trading day.
        self.network = self._make_network()

        # Store the selected trained policy for each risk objective and replicate.
        self.models = {}

        # Store datasets, such as training, validation, test, and calibration data.
        self.data = {}

        # Store each training run's history, selected checkpoint details, and scores.
        self.runs = []

        # Store numerical result arrays, such as each evaluated path's loss and cost.
        self.arrays = {}

        # Store summary scores for reference strategies, such as no hedge or delta hedging.
        self.benchmarks = {}

        # Record summaries of individual training batches, including losses and costs.
        self.batch_records = []

        # Store complete training batches when retain_training_batches is True.
        # Otherwise, the raw training batches are not kept here.
        self.training_batches = {}

        # Keep the evaluation results each time validate() is called.
        self.measurements = []

        # Store named results from evaluating policies on supplied historical data.
        self.historical_results = {}

        # Store descriptive information about the experiment.
        self.metadata = {

            # Name of the numerical scheme used by the built-in market simulator.
            "simulation_scheme": SIMULATION_SCHEME,

            # Identify this object as a newly created experiment.
            "origin": "New experiment",

            # Record whether training uses fresh batches or a fixed dataset.
            "batch_mode": "dynamic" if dynamic_data else "static",

            # Describe the loss convention and which trading fees are included.
            "accounting": "Loss before premium; opening and rebalancing fees; no maturity fee.",
        }
        if training_data is not None:
            if dynamic_data:
                raise ValueError("Use dynamic_data=False with a fixed training dataset.")
            self.load_data(training_data, "training")

    @property
    def parameter_count(self):
        """See title"""
        return sum(parameter.numel() for parameter in self.network.parameters())

    def _make_network(self, replicate=0):
        """"This function creates a new, untrained trading policy with reproducible initial weights"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings.initialization_seed + replicate)
            return _TradingPolicy(self.settings, self.method)

    def _check_data(self, data):
        """"Checks that the data in the right format, with the right number of days, stocks, and instruments for this experiment."""
        if isinstance(data, dict):
            data = DataSet(**data)
        if not isinstance(data, DataSet):
            raise TypeError("Supply a DataSet or a dictionary of its fields.")
        settings = self.settings
        if data.prices.shape[1:] != (settings.number_of_days + 1, settings.output_size):
            raise ValueError("Prices do not match the configured horizon and instruments.")
        if data.stock_prices.shape[-1] != len(settings.stocks):
            raise ValueError("Stock states do not match the configured modeled stocks.")
        for field_name, configured in (("instrument_names", settings.instruments), ("stock_names", settings.stocks)):
            names = getattr(data, field_name)
            if names is not None and list(names) != [item.name for item in configured]:
                raise ValueError(f"{field_name} must match the configured order.")
        return data

    def _check_separation(self, data, split):
        """"Checks that a new dataset is separate from training, validation, test, or historical data already in the experiment. It rejects reused seeds, identical simulated scenarios, or overlapping date ranges. """
        known_seeds = {self.settings.training_seed + 100 * replicate + stage
                       for replicate in range(self.settings.number_of_replicates) for stage in (0, 1)}
        if split in {"test", "historical"}:
            known_seeds.add(self.settings.validation_seed)
        if split == "validation":
            known_seeds.add(self.settings.test_seed)
        if split == "training":
            known_seeds = {self.settings.validation_seed, self.settings.test_seed}
        if data.seed is not None and data.seed in known_seeds:
            raise ValueError("This dataset uses a seed reserved for another split.")
        for name, known in self.data.items():
            if name == split or name == "example" or (split == "historical" and name == "test"):
                continue
            if data.fingerprint(True) == known.fingerprint(True):
                raise ValueError("Training, validation and evaluation data cannot be reused across splits.")
            if data.dates is not None and known.dates is not None:
                dates = np.asarray(data.dates, dtype="datetime64[ns]")
                other = np.asarray(known.dates, dtype="datetime64[ns]")
                if dates.min() <= other.max() and other.min() <= dates.max():
                    raise ValueError("Dated training and evaluation periods overlap.")

    def load_data(self, data, split="validation"):
        """Store explicitly supplied training, validation or test observations."""
        if split not in {"training", "validation", "test"}:
            raise ValueError("Choose training, validation or test.")
        if split == "training" and self.dynamic_data:
            raise ValueError("Set dynamic_data=False to train from a fixed dataset.")
        data = self._check_data(data)
        self._check_separation(data, split)
        data.role = split
        self.data[split] = data
        if split == "test":
            self.metadata["test_source"] = "Supplied by caller; known reuse rejected"
        return self

    def _generate(self, size, generator, seed, role):
        """This function is responsible for creating a 'DataSet' of market paths. """"
        if self.data_generator is None and self.metadata.get("custom_generator_requires_reconnection"):
            raise ValueError("Reconnect the saved experiment's custom data_generator before generating more data.")
        if self.data_generator is not None:
            data = self._check_data(self.data_generator(size, generator, self.settings))
            if len(data) != size:
                raise ValueError("The custom generator returned the wrong batch size.")
            data.role, data.seed = role, seed
            return data
        if any(item.kind == "observed" for item in self.settings.instruments):
            raise ValueError("Observed instruments require supplied data or an explicit data_generator.")
        stocks, variances, variance_claims = [], [], []
        payoff = np.zeros(size)
        for index, stock in enumerate(self.settings.stocks):
            prices, variance = simulate_paths(size, generator, self.settings.for_stock(index))
            stocks.append(prices[:, :, 0])
            variance_claims.append(prices[:, :, 1])
            variances.append(variance)
            payoff += stock.call_quantity * np.maximum(prices[:, -1, 0] - stock.strike, 0)
        instrument_prices = [
            stocks[item.stock_index] if item.kind == "stock" else variance_claims[item.stock_index]
            for item in self.settings.instruments
        ]
        return DataSet(np.stack(instrument_prices, axis=2), np.stack(stocks, axis=2),
                       np.stack(variances, axis=2), payoff, seed=seed, role=role,
                       instrument_names=[item.name for item in self.settings.instruments],
                       stock_names=[stock.name for stock in self.settings.stocks],
                       metadata={"kind": "synthetic", "stocks_independent": True})

    def prepare_data(self, include_test=True):
        """"Create the validation data, optionally create the test data, and create a fixed training dataset only when you are using static training. """
        torch.set_num_threads(self.settings.torch_threads)
        splits = [("validation", self.settings.validation_size, self.settings.validation_seed)]
        if not self.dynamic_data:
            splits.insert(0, ("training", self.settings.static_training_size, self.settings.training_seed))
        if include_test:
            splits.append(("test", self.settings.test_size, self.settings.test_seed))
        for split, size, seed in splits:
            if split not in self.data:
                self.data[split] = self._generate(size, np.random.default_rng(seed), seed, split)
                if split == "test":
                    self.metadata["test_source"] = "Generated independently by this experiment"
        return self

    def _batch(self, size, generator, seed, role):
        """Generates a batch. """

        if self.dynamic_data:
            # Generate a fresh batch of complete market paths.
            batch = self._generate(size, generator, seed, role)
            return batch, None

        else:
            # Get the fixed training dataset.
            pool = self.data["training"]

            # Randomly choose which paths to include in this batch.
            indices = generator.integers(0, len(pool), size=size)

            # Return the selected paths and their indices.
            batch = pool.select(indices, role)
            return batch, indices

    def calculate_losses(self, data, holdings):
        """Generic self-financing accounting, using the supplied payoff vector."""
        multipliers = data.prices.new_tensor([item.multiplier for item in self.settings.instruments])
        rates = data.prices.new_tensor([
            self.settings.cost_rate if item.cost_rate is None else item.cost_rate
            for item in self.settings.instruments
        ])
        quoted = data.prices * multipliers
        gains = (holdings * (quoted[:, 1:] - quoted[:, :-1] + data.cashflows * multipliers)).sum(dim=(1, 2))
        previous = torch.cat([torch.zeros_like(holdings[:, :1]), holdings[:, :-1]], dim=1)
        costs = (rates * quoted[:, :-1] * (holdings - previous).abs()).sum(dim=(1, 2))
        return data.payoffs - gains + costs, costs

    @torch.no_grad()
    def _score(self, model, data, batch_size=2048):
        was_training = model.training
        model.eval()
        losses, costs = [], []
        try:
            for first in range(0, len(data), batch_size):
                batch = data.select(np.arange(first, min(first + batch_size, len(data))))
                holdings = model(batch.stock_prices, batch.variances)
                loss, cost = self.calculate_losses(batch, holdings)
                losses.append(loss)
                costs.append(cost)
        finally:
            model.train(was_training)
        return torch.cat(losses), torch.cat(costs)

    @torch.no_grad()
    def predict(self, data, objective="CVaR", replicate=0):
        data = self._check_data(data)
        model = self.models[self.get_run(objective, replicate)["run_id"]] if self.runs else self.network
        was_training = model.training
        model.eval()
        try:
            return model(data.stock_prices, data.variances)
        finally:
            model.train(was_training)

    @torch.no_grad()
    def _calibrate(self, model, data):
        layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm1d)]
        if not layers:
            return
        def update_statistics(layer, inputs):
            layer.running_mean.copy_(inputs[0].mean(dim=0))
            layer.running_var.copy_(inputs[0].var(dim=0, unbiased=False))
        was_training = model.training
        model.eval()
        hooks = [layer.register_forward_pre_hook(update_statistics) for layer in layers]
        try:
            model(data.stock_prices, data.variances)
        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)

    def _train_stage(self, model, objective, replicate, name, seed, updates, learning_rate, verbose):
        settings = self.settings
        generator = np.random.default_rng(seed)
        calibration, _ = self._batch(settings.normalization_size, generator, seed, "calibration")
        stage = f"{self.family}_{replicate}_{name}"
        self.data["calibration_" + stage] = calibration
        self._calibrate(model, calibration)
        initial_losses, _ = self._score(model, calibration)
        initial = initial_losses.median() if objective == "CVaR" else risk_value(initial_losses, objective, settings)
        threshold = nn.Parameter(initial.detach().clone())
        parameters = list(model.parameters()) + [threshold]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        history = []
        best_risk = float("inf")
        clipped = 0
        start = time.perf_counter()
        for update in range(updates + 1):
            if update % settings.validation_interval == 0 or update == updates:
                self._calibrate(model, calibration)
                losses, _ = self._score(model, self.data["validation"])
                value = risk_value(losses, objective, settings).item()
                if not math.isfinite(value):
                    raise RuntimeError("Validation risk became nonfinite.")
                history.append({"update": update, "risk": value, "elapsed_seconds": time.perf_counter() - start})
                if value < best_risk:
                    best_risk, best_update = value, update
                    best_state, best_threshold = copy.deepcopy(model.state_dict()), threshold.item()
                if verbose and (update % 500 == 0 or update == updates):
                    print(f"{stage}: update {update}, validation risk {value:.5f}", flush=True)
            if update == updates:
                break
            batch, indices = self._batch(settings.batch_size, generator, seed, "training_batch")
            model.train()
            holdings = model(batch.stock_prices, batch.variances)
            losses, costs = self.calculate_losses(batch, holdings)
            loss = training_objective(losses, objective, threshold, settings)
            if not torch.isfinite(loss):
                raise RuntimeError("Training objective became nonfinite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(parameters, settings.gradient_limit, error_if_nonfinite=True)
            optimizer.step()
            clipped += int(gradient > settings.gradient_limit)
            record = {"stage": stage, "objective": objective, "replicate": replicate, "seed": seed,
                      "update": update + 1, "mode": "dynamic" if self.dynamic_data else "static",
                      "paths": len(batch), "training_objective": loss.item(), "mean_loss": losses.mean().item(),
                      "mean_cost": costs.mean().item(), "gradient_norm_before_clipping": gradient.item()}
            if indices is not None:
                record["pool_size"] = len(self.data["training"])
                record["distinct_pool_rows"] = len(np.unique(indices))
            if settings.retain_training_batches:
                self.training_batches.setdefault(stage, []).append(batch)
                record.update(losses=losses.detach().tolist(), costs=costs.detach().tolist())
                if indices is not None:
                    record["pool_indices"] = indices.tolist()
            self.batch_records.append(record)
        model.load_state_dict(best_state)
        model.eval()
        return {"history": history, "best_update": best_update, "best_validation_risk": best_risk,
                "threshold": best_threshold, "seconds": time.perf_counter() - start,
                "clipped_updates": clipped, "learning_rate": learning_rate, "training_seed": seed,
                "batch_mode": "dynamic" if self.dynamic_data else "static"}

    def train(self, objectives=OBJECTIVES, verbose=False):
        if self.runs:
            raise RuntimeError("Create a new generalNN object to run another training experiment.")
        objectives = tuple(objectives)
        if not objectives or len(set(objectives)) != len(objectives) or any(item not in OBJECTIVES for item in objectives):
            raise ValueError("Choose one or both distinct objectives: CVaR, Entropic.")
        self.prepare_data(include_test=False)
        settings = self.settings
        for replicate in range(settings.number_of_replicates):
            model = self._make_network(replicate)
            seed = settings.training_seed + 100 * replicate
            warmup = self._train_stage(model, "CVaR", replicate, "warmup", seed,
                                      settings.warmup_updates, settings.learning_rate, verbose)
            warm_state = copy.deepcopy(model.state_dict())
            for objective in objectives:
                model.load_state_dict(warm_state)
                rate = settings.learning_rate if objective == "CVaR" else settings.entropic_learning_rate
                stage = self._train_stage(model, objective, replicate, objective, seed + 1,
                                          settings.continuation_updates, rate, verbose)
                run_id = f"{self.family}_{objective}_{replicate}"
                self.runs.append({"run_id": run_id, "loss_key": "loss_" + run_id,
                                  "family": self.family, "objective": objective, "replicate_id": replicate,
                                  "simulation_scheme": SIMULATION_SCHEME, "parameter_count": self.parameter_count,
                                  "warmup": warmup, **stage})
                self.models[run_id] = copy.deepcopy(model)
        self.network = self.models[self.runs[0]["run_id"]]
        return self

    fit = train

    def validate(self):
        """Measure the current policies on stored validation data without fitting."""
        if "validation" not in self.data:
            raise ValueError("Load or prepare a validation dataset first.")
        records = {}
        for name, model in (self.models or {"initial_network": self.network}).items():
            losses, costs = self._score(model, self.data["validation"])
            records[name] = {"metrics": summarize(losses, costs, self.settings),
                             "losses": losses.numpy(), "costs": costs.numpy()}
        self.measurements.append({"split": "validation", "results": records})
        return {name: record["metrics"] for name, record in records.items()}

    def _evaluate_data(self, data):
        if not self.runs:
            raise RuntimeError("Train or load selected policies before evaluating them.")
        self.arrays = {}
        for report in self.runs:
            losses, costs = self._score(self.models[report["run_id"]], data)
            self.arrays[report["loss_key"]] = losses.numpy()
            self.arrays["costs_" + report["run_id"]] = costs.numpy()
            report["metrics"] = summarize(losses, costs, self.settings)
            report["diagnostics"] = {"all_losses_finite": True, "batch_normalization_frozen_for_test": True}
        self.arrays["benchmark_loss_no_hedge"] = data.payoffs.numpy()
        self.benchmarks = {"no_hedge": summarize(data.payoffs, torch.zeros_like(data.payoffs), self.settings)}
        self.arrays["example_prices"] = data.stock_prices[:6].numpy().copy()
        self.arrays["example_variances"] = data.variances[:6].numpy().copy()
        self.arrays["example_payoffs"] = data.payoffs[:6].numpy().copy()
        self.metadata["default_call_payoff"] = (
            data.metadata.get("kind") == "synthetic" and len(self.settings.stocks) == 1
            and torch.equal(data.payoffs, torch.relu(data.stock_prices[:, -1, 0] - self.settings.strike))
        )
        self.metadata.update(test_data_id=data.fingerprint(True), test_size=len(data), test_seed=data.seed)
        self.metadata["instrument_data_ids"] = {
            item.name: hashlib.sha256(data.prices[:, :, index].numpy().tobytes()
                                      + data.cashflows[:, :, index].numpy().tobytes()).hexdigest()
            for index, item in enumerate(self.settings.instruments)
        }
        return self

    def evaluate(self, data=None, include_delta=False):
        if self.metadata.get("evaluation_kind") == "historical":
            if data is not None or include_delta:
                raise ValueError("Use evaluate_historical for new historical observations; the analytical delta benchmark is synthetic.")
            return self._evaluate_data(self.data["historical"])
        if data is not None:
            self.load_data(data, "test")
        elif "test" not in self.data:
            self.prepare_data()
        self._evaluate_data(self.data["test"])
        if include_delta:
            self._add_delta_benchmark(self.data["test"])
        return self

    def _check_historical(self, data):
        if isinstance(data, dict) and data.get("metadata", {}).get("asset_cashflows") == "included" and "cashflows" not in data:
            raise ValueError("Supply the observed asset cashflows declared in the metadata.")
        data = self._check_data(data)
        metadata = data.metadata
        if metadata.get("kind") != "historical":
            raise ValueError("Explicitly identify the supplied observations as historical.")
        for key in ("source", "payoff_source", "variance_source"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Historical data requires {key} provenance.")
        if metadata.get("price_basis") != "raw_per_unit" or metadata.get("interest_rate") != 0:
            raise ValueError("Use raw per-unit prices and explicitly declare zero interest.")
        if metadata.get("liability_timing") != "terminal" or metadata.get("asset_cashflows") not in {"none", "included"}:
            raise ValueError("Declare terminal liability payments and whether asset cashflows are included.")
        if metadata["asset_cashflows"] == "none" and torch.any(data.cashflows != 0):
            raise ValueError("Cashflow values disagree with the declared convention.")
        if data.instrument_names is None or data.stock_names is None:
            raise ValueError("Historical data requires ordered stock and instrument names.")
        if data.dates is None or data.feature_times is None:
            raise ValueError("Supply execution dates and the availability times of the input features.")
        dates = np.asarray(data.dates, dtype="datetime64[ns]")
        available = np.asarray(data.feature_times, dtype="datetime64[ns]")
        if dates.shape != data.prices.shape[:2] or available.shape != data.prices[:, :-1].shape[:2]:
            raise ValueError("Historical dates or feature times have the wrong shape.")
        if np.isnat(dates).any() or np.isnat(available).any() or np.any(np.diff(dates, axis=1) <= np.timedelta64(0, "ns")):
            raise ValueError("Use finite, strictly increasing execution dates on every path.")
        if np.any(available > dates[:, :-1]):
            raise ValueError("A feature cannot become available after the trading decision.")
        self._check_separation(data, "historical")
        return data

    def evaluate_historical(self, data, name="Historical evaluation"):
        """Return a separate, plottable result using supplied historical payments.

        This never simulates missing prices, computes a replacement option
        payoff, fits weights, or calibrates normalization on evaluation data.
        """
        data = self._check_historical(data)
        result = copy.copy(self)
        result.name = name
        result.data = {"historical": data}
        result.runs = copy.deepcopy(self.runs)
        result.metadata = copy.deepcopy(self.metadata)
        result.metadata.update(evaluation_kind="historical", origin="Supplied historical observations",
                               historical_provenance=copy.deepcopy(data.metadata),
                               independent_historical_sample="Caller supplied; known reuse and date overlap rejected")
        result.historical_results = {}
        result._evaluate_data(data)
        self.historical_results[name] = result
        return result

    def get_run(self, objective="CVaR", replicate=0):
        for report in self.runs:
            if report["objective"] == objective and report["replicate_id"] == replicate:
                return report
        raise KeyError(f"No {objective} run with replicate {replicate}.")

    def losses(self, objective="CVaR", replicate=0):
        return self.arrays[self.get_run(objective, replicate)["loss_key"]]

    def costs(self, objective="CVaR", replicate=0):
        return self.arrays["costs_" + self.get_run(objective, replicate)["run_id"]]

    @classmethod
    def from_saved(cls, family="ours", folder=None):
        """Read the unchanged benchmark, without training or regenerating data."""
        family = {"paper": "paper_full", "control": "paper_stock"}.get(family, family)
        folder = Path(folder) if folder is not None else PROJECT_ROOT / "results"
        summary = json.loads((folder / "summary.json").read_text())
        settings = Settings(**summary["config"], trade_variance=(family != "paper_stock"))
        result = cls("ours" if family == "ours" else "paper", settings)
        result.runs = [report for report in summary["replicates"] if report["family"] == family]
        if not result.runs:
            raise ValueError(f"No saved runs for {family}.")
        keys = {"example_prices", "example_variances", "benchmark_loss_no_hedge", "benchmark_loss_stock_delta"}
        for report in result.runs:
            keys.update([report["loss_key"], "costs_" + report["run_id"]])
            model = result._make_network(report["replicate_id"])
            path = folder / "runs" / f"{family}_{report['replicate_id']}" / f"{report['objective']}.pt"
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            result.models[report["run_id"]] = model
        with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
            result.arrays = {key: arrays[key] for key in keys}
        for report in result.runs:
            losses = result.arrays[report["loss_key"]]
            if losses.shape != (settings.test_size,) or not np.isfinite(losses).all():
                raise ValueError("Invalid preserved test outcomes.")
        result.network = result.models[result.runs[0]["run_id"]]
        result.benchmarks = summary["benchmarks"]
        result.metadata.update(origin="Preserved benchmark: common Heston simulation and training protocol",
            reference_label=summary["reference_label"], test_data_id=f"archived-common-heston-{settings.test_seed}-{settings.test_size}",
            test_seed=settings.test_seed, test_size=settings.test_size, raw_training_batches_available=False,
            analytic_heston_call_price=summary["analytic_heston_call_price"])
        return result

    def _snapshot(self):
        def encode(value):
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value.copy())
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value
        payload = {name: encode(getattr(self, name)) for name in (
            "method", "name", "dynamic_data", "runs", "arrays", "benchmarks", "metadata", "batch_records", "measurements")}
        payload.update(format_version=1, settings=self.settings.to_dict(), network=self.network.state_dict(),
                       models={name: model.state_dict() for name, model in self.models.items()},
                       data={name: encode(vars(data)) for name, data in self.data.items()},
                       training_batches={name: [encode(vars(data)) for data in batches]
                                         for name, batches in self.training_batches.items()},
                       historical_results={name: result._snapshot() for name, result in self.historical_results.items()},
                       custom_generator_required=(self.data_generator is not None or
                                                  self.metadata.get("custom_generator_requires_reconnection", False)))
        return payload

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self._snapshot(), folder / "experiment.pt")
        return folder

    @classmethod
    def _restore(cls, payload):
        if payload["format_version"] != 1:
            raise ValueError("Unsupported experiment format.")
        def decode(value):
            if isinstance(value, torch.Tensor):
                return value.numpy()
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value
        result = cls(payload["method"], Settings.from_dict(payload["settings"]),
                     payload["dynamic_data"], name=payload["name"])
        for name in ("runs", "arrays", "benchmarks", "metadata", "batch_records", "measurements"):
            setattr(result, name, decode(payload[name]))
        result.network.load_state_dict(payload["network"], strict=True)
        result.network.eval()
        for report in result.runs:
            model = result._make_network(report["replicate_id"])
            model.load_state_dict(payload["models"][report["run_id"]], strict=True)
            model.eval()
            result.models[report["run_id"]] = model
        result.data = {name: DataSet(**decode(data)) for name, data in payload["data"].items()}
        result.training_batches = {name: [DataSet(**decode(data)) for data in batches]
                                   for name, batches in payload["training_batches"].items()}
        result.historical_results = {name: cls._restore(data) for name, data in payload["historical_results"].items()}
        if payload["custom_generator_required"]:
            result.metadata["custom_generator_requires_reconnection"] = True
        return result

    @classmethod
    def load(cls, folder):
        return cls._restore(torch.load(Path(folder) / "experiment.pt", map_location="cpu", weights_only=True))

    def _add_delta_benchmark(self, data):
        if data.metadata.get("kind") == "historical":
            raise ValueError("The analytical Heston benchmark is for the synthetic model only.")
        holdings = torch.zeros((len(data), self.settings.number_of_days, self.settings.output_size), dtype=torch.float64)
        for stock_index, stock in enumerate(self.settings.stocks):
            indices = [index for index, item in enumerate(self.settings.instruments)
                       if item.kind == "stock" and item.stock_index == stock_index and item.tradable]
            if not indices:
                raise ValueError("The stock-delta benchmark needs a tradable stock for every liability.")
            index = indices[0]
            delta = delta_holdings(data.stock_prices[:, :, stock_index], data.variances[:, :, stock_index],
                                   self.settings.for_stock(stock_index))
            holdings[:, :, index] = stock.call_quantity * delta / self.settings.instruments[index].multiplier
        losses, costs = self.calculate_losses(data, holdings)
        self.arrays["benchmark_loss_stock_delta"] = losses.numpy()
        self.benchmarks["stock_delta"] = summarize(losses, costs, self.settings)


# Numerical helpers: the existing simulator, risk formulas and delta benchmark.

try:
    from numba import njit
except ImportError:
    njit = None


PATHS_PER_CHUNK = 512
SIMULATION_SCHEME = "exact_CIR_endpoints_mean_preserving_integral_stock_martingale_correction"
SIMULATION_BACKEND = "numba" if njit is not None else "numpy"


def variance_claim_price(integrated_variance, variance, remaining_time, settings):
    """Conditional continuous-Heston price of the claim paying integrated variance."""
    loading = -np.expm1(-settings.mean_reversion * np.asarray(remaining_time)) / settings.mean_reversion
    return (np.asarray(integrated_variance)
            + (np.asarray(variance)-settings.long_run_variance)*loading
            + settings.long_run_variance*np.asarray(remaining_time))


def transition_parameters(settings):
    """Return the exact CIR transition and the two martingale-correction weights."""
    dt = settings.time_step/settings.simulation_substeps
    kappa = settings.mean_reversion
    theta = settings.long_run_variance
    xi = settings.vol_of_vol
    rho = settings.correlation
    if min(dt, kappa, theta, xi) <= 0 or not -1 <= rho <= 1:
        raise ValueError("Use positive Heston time/variance scales and correlation in [-1, 1].")
    x = kappa*dt
    decay = math.exp(-x)
    one_minus_decay = -math.expm1(-x)
    loading = one_minus_decay/kappa
    # b = (dt - loading)/(1 - decay); its series avoids small-step cancellation.
    if abs(x) < 1e-3:
        next_weight = dt*(0.5 + x/12 - x**3/720 + x**5/30240)
    else:
        next_weight = (dt-loading)/one_minus_decay
    old_weight = dt-next_weight
    cir_scale = xi**2*one_minus_decay/(4*kappa)
    degrees = 4*kappa*theta/xi**2
    exponent = rho/xi + (rho*kappa/xi-0.5*rho**2)*next_weight
    denominator = 1-2*cir_scale*exponent
    if min(old_weight, next_weight, denominator) <= 0:
        raise ValueError("The simulation step does not satisfy the positive-weight/CIR-MGF conditions.")
    return {
        "dt": dt, "decay": decay, "variance_loading": loading,
        "old_variance_weight": old_weight, "next_variance_weight": next_weight,
        "cir_scale": cir_scale, "cir_degrees": degrees,
        "noncentrality_multiplier": decay/cir_scale,
        "stock_exponent": exponent,
        "log_mgf_constant": -0.5*degrees*math.log1p(-2*cir_scale*exponent),
        "log_mgf_variance_coefficient": decay*exponent/denominator,
        "independent_variance_fraction": 1-rho**2,
    }


def _simulate_arrays(number_of_paths, generator, initial_price, initial_variance,
                     number_of_days, simulation_substeps, coefficients):
    (degrees, cir_scale, noncentrality_multiplier, old_weight, next_weight,
     stock_exponent, mgf_constant, mgf_variance_coefficient, independent_fraction) = coefficients
    stock = np.full(number_of_paths, initial_price, dtype=np.float64)
    variance = np.full(number_of_paths, initial_variance, dtype=np.float64)
    next_variance = np.empty(number_of_paths, dtype=np.float64)
    integrated_variance = np.zeros(number_of_paths, dtype=np.float64)
    shape = (number_of_paths, number_of_days+1)
    stocks = np.empty(shape, dtype=np.float64)
    variances = np.empty(shape, dtype=np.float64)
    integrals = np.empty(shape, dtype=np.float64)
    stocks[:, 0] = stock
    variances[:, 0] = variance
    integrals[:, 0] = integrated_variance

    for day in range(number_of_days):
        for substep in range(simulation_substeps):
            for path in range(number_of_paths):
                noncentrality = variance[path]*noncentrality_multiplier
                next_variance[path] = cir_scale*generator.noncentral_chisquare(degrees, noncentrality)
            increment = old_weight*variance + next_weight*next_variance
            independent_variance = independent_fraction*increment
            log_mgf = mgf_constant + mgf_variance_coefficient*variance
            noise = generator.standard_normal(number_of_paths)
            log_return = (stock_exponent*next_variance - log_mgf - 0.5*independent_variance
                          + np.sqrt(independent_variance)*noise)
            stock *= np.exp(log_return)
            integrated_variance += increment
            variance, next_variance = next_variance, variance
        stocks[:, day+1] = stock
        variances[:, day+1] = variance
        integrals[:, day+1] = integrated_variance
    return stocks, variances, integrals


_simulate_fast = njit(cache=True)(_simulate_arrays) if njit is not None else _simulate_arrays


def _from_generator(number_of_paths, generator, settings, use_numba=True):
    simulator = _simulate_fast if use_numba else _simulate_arrays
    parameters = transition_parameters(settings)
    keys = ["cir_degrees", "cir_scale", "noncentrality_multiplier", "old_variance_weight",
            "next_variance_weight", "stock_exponent", "log_mgf_constant",
            "log_mgf_variance_coefficient", "independent_variance_fraction"]
    coefficients = tuple(parameters[key] for key in keys)
    return simulator(number_of_paths, generator, settings.initial_price, settings.initial_variance,
                     settings.number_of_days, settings.simulation_substeps, coefficients)


def _package(stocks, variances, integrals, settings, return_diagnostics):
    remaining = (settings.number_of_days-np.arange(settings.number_of_days+1))*settings.time_step
    claim = variance_claim_price(integrals, variances, remaining, settings)
    prices = np.stack([stocks, claim], axis=-1)
    if return_diagnostics:
        return prices, variances, {
            "integrated_variance": integrals,
            "raw_variance": variances,
            "simulation_scheme": SIMULATION_SCHEME,
            "simulation_backend": SIMULATION_BACKEND,
        }
    return prices, variances


def simulate_paths(number_of_paths, generator, settings, *, return_diagnostics=False, use_numba=True):
    """Return float64 NumPy asset prices (N, days+1, 2) and variances (N, days+1).

    Asset order is stock, raw variance claim. All randomness comes from the
    supplied NumPy Generator. The optional third return value contains daily
    integrated variance and a raw-variance alias (no truncation is required).
    Set use_numba=False to run the same equations in ordinary NumPy/Python.
    """
    if number_of_paths < 1 or settings.simulation_substeps < 1:
        raise ValueError("Path count and simulation substeps must be positive.")
    shape = (number_of_paths, settings.number_of_days+1)
    stocks = np.empty(shape, dtype=np.float64)
    variances = np.empty(shape, dtype=np.float64)
    integrals = np.empty(shape, dtype=np.float64)
    for first in range(0, number_of_paths, PATHS_PER_CHUNK):
        last = min(first+PATHS_PER_CHUNK, number_of_paths)
        stocks[first:last], variances[first:last], integrals[first:last] = _from_generator(
            last-first, generator, settings, use_numba)
    return _package(stocks, variances, integrals, settings, return_diagnostics)


def cvar(losses, alpha=0.5):
    rank = max(1, math.ceil(alpha * len(losses)))
    threshold = losses.kthvalue(rank).values
    return threshold + torch.relu(losses - threshold).mean() / (1 - alpha)


def entropic_risk(losses, risk_aversion=1.0):
    log_average = torch.logsumexp(risk_aversion * losses, dim=0) - math.log(len(losses))
    return log_average / risk_aversion


def risk_value(losses, objective, settings):
    if objective == "CVaR":
        return cvar(losses, settings.alpha)
    if objective == "Entropic":
        return entropic_risk(losses, settings.risk_aversion)
    raise ValueError("Choose CVaR or Entropic.")


def training_objective(losses, objective, threshold, settings):
    if objective == "CVaR":
        return threshold + torch.relu(losses - threshold).mean() / (1 - settings.alpha)
    if objective == "Entropic":
        exponent = settings.risk_aversion * (losses - threshold)
        return threshold + torch.expm1(exponent).mean() / settings.risk_aversion
    raise ValueError("Choose CVaR or Entropic.")


def summarize(losses, costs, settings):
    weights = torch.softmax(settings.risk_aversion * losses, dim=0)
    return {
        "mean_loss": losses.mean().item(),
        "cvar": cvar(losses, settings.alpha).item(),
        "entropic": entropic_risk(losses, settings.risk_aversion).item(),
        "mean_cost": costs.mean().item(),
        "max_loss": losses.max().item(),
        "entropic_effective_paths": (1 / weights.square().sum()).item(),
        "entropic_max_weight": weights.max().item(),
    }


from scipy.fft import dst
from scipy.integrate import quad


def _characteristic(frequency, maturity, variance, settings):
    """Characteristic function of log(S_T/S_t), with the stable Heston branch."""
    kappa = settings.mean_reversion
    theta = settings.long_run_variance
    xi = settings.vol_of_vol
    rho = settings.correlation
    u = np.asarray(frequency, dtype=complex)
    beta = kappa - 1j * rho * xi * u
    root = np.sqrt(beta**2 + xi**2 * (u**2 + 1j*u))
    ratio = (beta - root) / (beta + root)
    exponential = np.exp(-root * maturity)
    log_ratio = np.log1p(-ratio * exponential) - np.log1p(-ratio)
    constant = kappa * theta / xi**2 * ((beta - root) * maturity - 2 * log_ratio)
    coefficient = (beta - root) / xi**2
    coefficient *= -np.expm1(-root * maturity) / (1 - ratio * exponential)
    return np.exp(constant + coefficient * variance)


def _validate_parameters(settings):
    if min(settings.mean_reversion, settings.long_run_variance, settings.vol_of_vol) <= 0:
        raise ValueError("The Heston benchmark requires positive mean reversion, long-run variance, and vol of vol.")
    if not -1 < settings.correlation < 1:
        raise ValueError("The Heston benchmark requires correlation strictly between -1 and 1.")


def call_price(settings, *, spot=None, variance=None, maturity=None):
    """Return the zero-interest European call price from Fourier probabilities."""
    _validate_parameters(settings)
    spot = settings.initial_price if spot is None else float(spot)
    variance = settings.initial_variance if variance is None else float(variance)
    maturity = settings.maturity if maturity is None else float(maturity)
    if spot <= 0 or variance < 0 or maturity <= 0 or settings.strike <= 0:
        raise ValueError("Use positive stock price, strike and maturity, and nonnegative variance.")
    log_moneyness = math.log(spot / settings.strike)
    probabilities = []

    for shift in [1.0, 0.0]:
        def integrand(frequency):
            value = np.exp(1j * frequency * log_moneyness)
            value *= _characteristic(frequency - 1j*shift, maturity, variance, settings)
            return value.imag / frequency

        integral, error = quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=2000)
        if not math.isfinite(integral) or error > 1e-7:
            raise RuntimeError("Heston call-price integration did not meet its error tolerance.")
        probabilities.append(0.5 + integral / math.pi)
    return spot * probabilities[0] - settings.strike * probabilities[1]


def _delta_curve(variance, maturity, settings, domain_factor, resolution):
    """Compute a dense delta curve in log-moneyness for one variance and date."""
    kappa = settings.mean_reversion
    theta = settings.long_run_variance
    xi = settings.vol_of_vol
    rho = settings.correlation
    expected_integral = theta*maturity + (variance-theta)*(-math.expm1(-kappa*maturity))/kappa
    gaussian_width = math.sqrt(max(expected_integral, 1e-20))
    fourier_width = (variance + kappa*theta*maturity) * math.sqrt(1-rho**2) / xi
    narrow_width = min(gaussian_width, fourier_width)
    half_width = domain_factor * (math.sqrt(variance*maturity) + xi*maturity) + expected_integral
    requested = max(2048, 2 * half_width / narrow_width * resolution)
    number = 2**int(math.ceil(math.log2(requested)))
    if number > 2**20:
        raise RuntimeError("These parameters need a larger Heston benchmark grid than the supported budget.")

    indices = np.arange(1, number)
    frequencies = indices * math.pi / (2 * half_width)
    # Under the stock numeraire, E[exp(i*u*log-return)] = phi(u-i).
    tilted_characteristic = _characteristic(frequencies - 1j, maturity, variance, settings)
    cosine_coefficients = np.real(tilted_characteristic * np.exp(1j * frequencies * half_width))
    sine_coefficients = cosine_coefficients / (indices * math.pi)
    cdf = np.empty(number + 1)
    cdf[0], cdf[-1] = 0.0, 1.0
    cdf[1:-1] = indices / number + dst(sine_coefficients, type=1)
    bound_error = max(0.0, -float(cdf.min()), float(cdf.max()) - 1.0)
    monotonicity_error = max(0.0, -float(np.diff(cdf).min()))
    if max(bound_error, monotonicity_error) > 1e-7:
        raise RuntimeError("The Heston delta CDF needs a finer Fourier grid.")
    cdf = np.clip(cdf, 0.0, 1.0)
    log_return = np.linspace(-half_width, half_width, number + 1)
    log_moneyness = -log_return[::-1]
    delta = 1.0 - cdf[::-1]
    diagnostics = {
        "fourier_nodes": number,
        "cdf_bound_error": bound_error,
        "cdf_monotonicity_error": monotonicity_error,
    }
    return log_moneyness, delta, diagnostics


def delta_holdings(prices, variances, settings, *, return_diagnostics=False,
                   variance_step=0.06, domain_factor=8.0, resolution=16.0):
    """Return Heston stock deltas at the daily trading dates.

    Supply stock and nonnegative effective-variance paths with shape
    (number_of_paths, number_of_days + 1). No training or saving occurs.
    The benchmark uses a fixed grid in log(1 + v/(kappa*theta*tau)); it computes
    only the curves needed by the supplied states. Adding other paths does not
    change a given state's grid or its result. Curve boundaries approximate
    remote-tail probabilities by zero and one; diagnostics count these uses.
    """
    _validate_parameters(settings)
    start = time.perf_counter()
    stock_array = prices.detach().cpu().numpy()
    variance_array = variances.detach().cpu().numpy()
    expected_columns = settings.number_of_days + 1
    if stock_array.ndim != 2 or stock_array.shape != variance_array.shape or stock_array.shape[1] != expected_columns:
        raise ValueError("Price and variance paths must have matching (paths, days + 1) shapes.")
    if not np.isfinite(stock_array).all() or not np.isfinite(variance_array).all():
        raise ValueError("Price and variance paths must be finite.")
    if np.any(stock_array <= 0) or np.any(variance_array < 0):
        raise ValueError("Use positive prices and nonnegative effective variances.")
    if min(variance_step, domain_factor, resolution) <= 0:
        raise ValueError("The interpolation and Fourier settings must be positive.")

    holdings = np.zeros((len(stock_array), settings.number_of_days))
    diagnostics = {
        "method": "Analytical Heston stock delta via Fourier-COS CDF and log-variance interpolation",
        "variance_step": variance_step,
        "domain_factor": domain_factor,
        "resolution": resolution,
        "curves_computed": 0,
        "maximum_fourier_nodes": 0,
        "maximum_cdf_bound_error": 0.0,
        "maximum_cdf_monotonicity_error": 0.0,
        "remote_tail_interpolations": 0,
    }

    for day in range(settings.number_of_days):
        maturity = (settings.number_of_days - day) * settings.time_step
        log_moneyness = np.log(stock_array[:, day] / settings.strike)
        variance = variance_array[:, day]
        scale = settings.mean_reversion * settings.long_run_variance * maturity
        grid_position = np.log1p(variance / scale) / variance_step
        lower = np.floor(grid_position).astype(int)
        fraction = grid_position - lower
        upper = lower + 1
        needed_nodes = np.unique(np.concatenate([lower, upper[fraction > 0]]))

        for node in needed_nodes:
            grid_variance = scale * math.expm1(float(node) * variance_step)
            x_grid, delta_grid, information = _delta_curve(
                grid_variance, maturity, settings, domain_factor, resolution)
            diagnostics["curves_computed"] += 1
            diagnostics["maximum_fourier_nodes"] = max(diagnostics["maximum_fourier_nodes"], information["fourier_nodes"])
            diagnostics["maximum_cdf_bound_error"] = max(diagnostics["maximum_cdf_bound_error"], information["cdf_bound_error"])
            diagnostics["maximum_cdf_monotonicity_error"] = max(diagnostics["maximum_cdf_monotonicity_error"], information["cdf_monotonicity_error"])

            for mask, weight in [(lower == node, 1 - fraction), ((upper == node) & (fraction > 0), fraction)]:
                queries = log_moneyness[mask]
                interpolated = np.interp(queries, x_grid, delta_grid)
                holdings[mask, day] += weight[mask] * interpolated
                diagnostics["remote_tail_interpolations"] += int(np.count_nonzero(
                    (queries < x_grid[0]) | (queries > x_grid[-1])))

    diagnostics["seconds"] = time.perf_counter() - start
    result = torch.as_tensor(holdings, dtype=prices.dtype, device=prices.device)
    return (result, diagnostics) if return_diagnostics else result
