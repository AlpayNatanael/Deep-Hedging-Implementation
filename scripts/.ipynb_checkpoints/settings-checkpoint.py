"""All choices for networks, modeled stocks, trading instruments and training."""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields, replace
from types import SimpleNamespace
import math

FAMILIES = {
    "ours": "Our network: stock only",
    "paper_full": "Paper style: stock + variance",
    "paper_stock": "Paper style: stock only",
}
OBJECTIVES = ("CVaR", "Entropic")


@dataclass(frozen=True)
class NetworkSettings:
    hidden_sizes: tuple = (16, 16)
    activation: str = "tanh"
    batch_normalization: bool = False

    def validate(self):
        if len(self.hidden_sizes) != 2 or any(type(size) is not int or size < 1 for size in self.hidden_sizes):
            raise ValueError("Keep two hidden layers with positive integer widths.")
        if self.activation not in {"tanh", "relu"}:
            raise ValueError("Choose tanh or relu.")
        if not isinstance(self.batch_normalization, bool):
            raise TypeError("batch_normalization must be a boolean.")


@dataclass(frozen=True)
class StockSettings:
    name: str = "Stock"
    initial_price: float = 100.0
    strike: float = 100.0
    initial_variance: float = 0.04
    long_run_variance: float = 0.04
    mean_reversion: float = 1.0
    vol_of_vol: float = 2.0
    correlation: float = -0.7
    call_quantity: float = 1.0


@dataclass(frozen=True)
class InstrumentSettings:
    name: str = "Stock"
    kind: str = "stock"
    stock_index: int = 0
    multiplier: float = 1.0
    tradable: bool = True
    cost_rate: float | None = None


@dataclass(frozen=True)
class Settings:
    initial_price: float = 100.0
    strike: float = 100.0
    initial_variance: float = 0.04
    long_run_variance: float = 0.04
    mean_reversion: float = 1.0
    vol_of_vol: float = 2.0
    correlation: float = -0.7
    number_of_days: int = 30
    time_step: float = 1.0 / 365.0
    simulation_substeps: int = 64
    cost_rate: float = 0.001
    variance_notional: float = 1000.0
    alpha: float = 0.5
    risk_aversion: float = 1.0
    warmup_updates: int = 3000
    continuation_updates: int = 3000
    learning_rate: float = 0.005
    entropic_learning_rate: float = 0.0005
    gradient_limit: float = 10.0
    batchnorm_epsilon: float = 0.001
    batchnorm_momentum: float = 0.01
    normalization_size: int = 8192
    batch_size: int = 256
    validation_size: int = 8192
    test_size: int = 65536
    validation_interval: int = 100
    number_of_replicates: int = 3
    display_replicate: int = 0
    initialization_seed: int = 7000
    training_seed: int = 10100
    validation_seed: int = 20200
    test_seed: int = 40400
    torch_threads: int = 1

    static_training_size: int = 32768
    retain_training_batches: bool = False
    network: NetworkSettings | None = None
    stocks: tuple | None = None
    instruments: tuple | None = None
    trade_variance: bool = True
    method: str | None = None

    @property
    def maturity(self):
        return self.number_of_days * self.time_step

    def _validate_numerics(self):
        positive = [
            self.initial_price, self.strike, self.initial_variance,
            self.long_run_variance, self.mean_reversion, self.vol_of_vol,
            self.number_of_days, self.time_step, self.simulation_substeps,
            self.variance_notional, self.risk_aversion, self.learning_rate,
            self.entropic_learning_rate, self.gradient_limit, self.batch_size,
            self.batchnorm_epsilon, self.batchnorm_momentum,
            self.normalization_size,
            self.validation_size, self.test_size, self.validation_interval,
            self.number_of_replicates, self.torch_threads,
        ]
        if min(positive) <= 0:
            raise ValueError("Sizes and model scales must be positive.")
        if min(self.warmup_updates, self.continuation_updates, self.cost_rate) < 0:
            raise ValueError("Update counts and transaction costs cannot be negative.")
        if not 0 <= self.alpha < 1 or not -1 <= self.correlation <= 1:
            raise ValueError("Invalid CVaR level or Brownian correlation.")
        if not 0 <= self.display_replicate < self.number_of_replicates:
            raise ValueError("The displayed replicate must be a configured replicate.")
        if len({self.training_seed, self.validation_seed, self.test_seed}) != 3:
            raise ValueError("Training, validation and test seeds must be different.")

    @property
    def input_size(self):
        return 2 * len(self.stocks) + len(self.instruments)

    @property
    def output_size(self):
        return len(self.instruments)

    def resolve(self, method):
        """Choose a preset; an explicit network or instrument list takes priority."""
        if method not in {"ours", "paper"}:
            raise ValueError("The required method flag must be 'ours' or 'paper'.")
        family = "ours" if method == "ours" else "paper_full" if self.trade_variance else "paper_stock"
        if self.method is not None and self.method != family:
            raise ValueError("The method flag disagrees with these resolved settings.")
        names = [field.name for field in fields(StockSettings) if hasattr(self, field.name)]
        stocks = self.stocks
        if stocks is None:
            stocks = (StockSettings(**{name: getattr(self, name) for name in names}),)
        if not stocks:
            raise ValueError("Configure at least one modeled stock.")
        instruments = self.instruments
        if instruments is None:
            instruments = []
            for index, stock in enumerate(stocks):
                instruments.append(InstrumentSettings(stock.name, stock_index=index))
                if method == "paper":
                    instruments.append(InstrumentSettings(
                        stock.name + " variance", "variance", index,
                        multiplier=self.variance_notional, tradable=self.trade_variance,
                    ))
        instruments = tuple(instruments)
        if method == "paper" and not self.trade_variance:
            instruments = tuple(replace(item, tradable=False) if item.kind == "variance" else item
                                for item in instruments)
        network = self.network
        if network is None:
            width = 16 if method == "ours" else len(instruments) + 15
            network = NetworkSettings((width, width), "tanh" if method == "ours" else "relu", method == "paper")
        first_stock = {name: getattr(stocks[0], name) for name in names}
        resolved = replace(self, method=family, stocks=tuple(stocks), instruments=instruments,
                           network=network, **first_stock)
        resolved.validate()
        return resolved

    def validate(self):
        self._validate_numerics()
        integer_names = ("number_of_days", "simulation_substeps", "normalization_size", "batch_size",
                         "validation_size", "test_size", "validation_interval", "number_of_replicates",
                         "torch_threads", "static_training_size")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in integer_names):
            raise ValueError("Sizes and simulation counts must be positive integers.")
        if any(type(value) is not int or value < 0 for value in (self.warmup_updates, self.continuation_updates)):
            raise ValueError("Update counts must be nonnegative integers.")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Numerical settings must be finite.")
        if not isinstance(self.retain_training_batches, bool) or not isinstance(self.trade_variance, bool):
            raise TypeError("Retention and trading flags must be booleans.")
        if not 0 < self.batchnorm_momentum <= 1:
            raise ValueError("Batch-normalization momentum must be in (0, 1].")
        training_seeds = {self.training_seed + 100 * replicate + stage
                          for replicate in range(self.number_of_replicates) for stage in (0, 1)}
        if training_seeds & {self.validation_seed, self.test_seed}:
            raise ValueError("Training-stage seeds must differ from held-out seeds.")
        if self.network is not None:
            self.network.validate()
            if self.network.batch_normalization and self.batch_size < 2:
                raise ValueError("Batch normalization needs at least two training paths.")
        if self.stocks is not None:
            if len({item.name for item in self.stocks}) != len(self.stocks):
                raise ValueError("Give the modeled stocks distinct names.")
            for stock in self.stocks:
                scales = (stock.initial_price, stock.strike, stock.initial_variance,
                          stock.long_run_variance, stock.mean_reversion, stock.vol_of_vol)
                if any(not math.isfinite(value) or value <= 0 for value in scales):
                    raise ValueError("Stock and Heston scales must be positive and finite.")
                if not -1 <= stock.correlation <= 1 or not math.isfinite(stock.call_quantity):
                    raise ValueError("Invalid stock correlation or liability quantity.")
        if self.instruments is not None:
            if not self.instruments or len({item.name for item in self.instruments}) != len(self.instruments):
                raise ValueError("Give the trading instruments distinct names.")
            for item in self.instruments:
                if item.kind not in {"stock", "variance", "observed"}:
                    raise ValueError("Instrument kind must be stock, variance or observed.")
                if type(item.stock_index) is not int or not 0 <= item.stock_index < len(self.stocks):
                    raise ValueError("An instrument refers to an unknown stock.")
                if not isinstance(item.tradable, bool) or not math.isfinite(item.multiplier) or item.multiplier <= 0:
                    raise ValueError("Instrument flags and price multipliers are invalid.")
                if item.cost_rate is not None and (not math.isfinite(item.cost_rate) or item.cost_rate < 0):
                    raise ValueError("Instrument costs must be finite and nonnegative.")

    def for_stock(self, index):
        values = self.to_dict()
        values.update(asdict(self.stocks[index]))
        values["maturity"] = self.maturity
        return SimpleNamespace(**values)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if isinstance(values.get("network"), dict):
            design = dict(values["network"])
            design["hidden_sizes"] = tuple(design["hidden_sizes"])
            values["network"] = NetworkSettings(**design)
        for name, record_type in (("stocks", StockSettings), ("instruments", InstrumentSettings)):
            if values.get(name) is not None:
                values[name] = tuple(record_type(**record) for record in values[name])
        return cls(**values)


ExperimentSettings = Settings
