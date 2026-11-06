"""Check the generalNN refactor without a large training experiment.

The manufactured historical fixture below is a software test, never real
market data or evidence of historical investment performance. Use --oracle
with the previous project to compare all saved files and seven figure renders.
"""

import argparse
import copy
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from scripts.general_nn import DataSet, generalNN
from scripts.settings import InstrumentSettings, NetworkSettings, Settings, StockSettings


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(function, message):
    try:
        function()
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError(message)


def canonical(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if is_dataclass(value):
        return canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def snapshot(experiment, include_historical=True):
    names = ("settings", "method", "name", "dynamic_data", "data", "runs",
             "arrays", "benchmarks", "batch_records", "training_batches",
             "measurements", "metadata")
    result = {name: canonical(getattr(experiment, name)) for name in names}
    result["network"] = canonical(experiment.network.state_dict())
    result["models"] = {name: canonical(model.state_dict())
                        for name, model in experiment.models.items()}
    if include_historical:
        result["historical_results"] = {name: snapshot(view, False)
                                         for name, view in experiment.historical_results.items()}
    return result


def tiny_settings(**changes):
    settings = Settings(number_of_days=3, simulation_substeps=8,
                        warmup_updates=1, continuation_updates=2,
                        batch_size=24, normalization_size=24,
                        validation_size=16, test_size=48, validation_interval=1,
                        number_of_replicates=1, static_training_size=7,
                        retain_training_batches=True, torch_threads=1)
    return replace(settings, **changes)


def data_rows(data):
    return {hashlib.sha256(data.stock_prices[index].numpy().tobytes()
                          + data.variances[index].numpy().tobytes()).hexdigest()
            for index in range(len(data))}


def independent_metrics(losses, costs, settings):
    losses = np.asarray(losses, dtype=float)
    ordered = np.sort(losses)[::-1]
    tail_size = (1 - settings.alpha) * len(losses)
    whole = int(np.floor(tail_size))
    total = ordered[:whole].sum()
    if whole < len(losses):
        total += (tail_size - whole) * ordered[whole]
    scaled = settings.risk_aversion * losses
    shift = scaled.max()
    exponentials = np.exp(scaled - shift)
    weights = exponentials / exponentials.sum()
    return {"mean_loss": float(losses.mean()), "cvar": float(total / tail_size),
            "entropic": float((shift + np.log(exponentials.mean())) / settings.risk_aversion),
            "mean_cost": float(np.mean(costs)), "max_loss": float(losses.max()),
            "entropic_effective_paths": float(1 / np.sum(weights**2)),
            "entropic_max_weight": float(weights.max())}


def cash_ledger(data, holdings, settings):
    """Independent cash/asset ledger, including interval payments and final mark."""
    multipliers = np.array([item.multiplier for item in settings.instruments])
    rates = np.array([settings.cost_rate if item.cost_rate is None else item.cost_rate
                      for item in settings.instruments])
    quotes = data.prices.numpy() * multipliers
    payments = data.cashflows.numpy() * multipliers
    holdings = np.asarray(holdings)
    cash = np.zeros(len(data))
    previous = np.zeros((len(data), len(multipliers)))
    total_cost = np.zeros(len(data))
    for day in range(settings.number_of_days):
        trades = holdings[:, day] - previous
        fee = np.sum(rates * quotes[:, day] * np.abs(trades), axis=1)
        cash -= np.sum(quotes[:, day] * trades, axis=1) + fee
        cash += np.sum(holdings[:, day] * payments[:, day], axis=1)
        total_cost += fee
        previous = holdings[:, day]
    terminal_assets = np.sum(previous * quotes[:, -1], axis=1)
    loss = data.payoffs.numpy() - cash - terminal_assets
    return loss, total_cost


def check_defaults_and_generic_policy():
    rejects(lambda: generalNN(), "The method flag unexpectedly became optional.")
    rejects(lambda: generalNN("unknown"), "An invalid method was accepted.")
    rejects(lambda: generalNN("ours", dynamic_data="False"), "A string was accepted as the data-mode flag.")
    rows = []
    for method, settings, count, shapes in (
        ("ours", Settings(), 10590, [(3, 16), (16, 16), (16, 1)]),
        ("paper", Settings(), 14850, [(4, 17), (17, 17), (17, 2)]),
        ("paper", Settings(trade_variance=False), 14850, [(4, 17), (17, 17), (17, 2)]),
    ):
        experiment = generalNN(method, settings)
        expected = ([nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
                    if method == "ours" else [nn.Linear, nn.BatchNorm1d, nn.ReLU,
                                              nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear])
        require(len(experiment.network.networks) == 30, "Default daily-network count changed.")
        for network in experiment.network.networks:
            require([type(layer) for layer in network] == expected, "Default layer order changed.")
            actual = [(layer.in_features, layer.out_features) for layer in network
                      if isinstance(layer, nn.Linear)]
            require(actual == shapes, "A default network shape changed.")
        parameters = list(experiment.network.parameters())
        require(experiment.parameter_count == count, "Default parameter count changed.")
        require(all(value.dtype == torch.float64 for value in parameters), "Parameter precision changed.")
        require(len({value.data_ptr() for value in parameters}) == len(parameters),
                "Daily parameters unexpectedly share storage.")
        rows.append({"family": experiment.family, "parameters": count, "shapes": shapes})

    stocks = (StockSettings("Alpha"), StockSettings("Beta", initial_price=50, strike=50))
    instruments = (InstrumentSettings("Alpha", stock_index=0),
                   InstrumentSettings("Beta", stock_index=1),
                   InstrumentSettings("Gamma", kind="observed", stock_index=0))
    settings = tiny_settings(stocks=stocks, instruments=instruments)
    for method in ("ours", "paper"):
        experiment = generalNN(method, settings)
        require(experiment.settings.input_size == 7 and experiment.settings.output_size == 3,
                "The generic state/holding dimensions are incorrect.")
        stock = torch.tensor([[[100., 50.], [102., 49.], [99., 52.], [104., 51.]]], dtype=torch.float64)
        variance = torch.full_like(stock, .04)
        experiment.network.eval()
        with torch.no_grad():
            before = experiment.network(stock, variance)
            altered = stock.clone()
            altered[:, 2:] *= 1.2
            future = experiment.network(altered, variance)
            require(torch.equal(before[:, :2], future[:, :2]), "Future prices changed earlier decisions.")
            for parameter in experiment.network.parameters():
                parameter.zero_()
            for network in experiment.network.networks:
                network[-1].bias.copy_(torch.tensor([.2, .3, .4], dtype=torch.float64))
            holdings = experiment.network(stock, variance)
        require(holdings.shape == (1, 3, 3), "Generic holdings have the wrong shape.")
        require(torch.allclose(holdings[0, 0], torch.tensor([.2, .3, .4], dtype=torch.float64)),
                "One of the generic outputs is silently suppressed.")

    control = generalNN("paper", tiny_settings(trade_variance=False))
    control.network.eval()
    with torch.no_grad():
        for network in control.network.networks:
            network[-1].bias[1] = 10.0
        holdings = control.network(torch.full((2, 4, 1), 100., dtype=torch.float64),
                                   torch.full((2, 4, 1), .04, dtype=torch.float64))
    require(torch.count_nonzero(holdings[:, :, 1]) == 0, "The legacy stock-only paper control can trade variance.")
    custom = generalNN("ours", tiny_settings(network=NetworkSettings((7, 5), "relu", True)))
    shapes = [(layer.in_features, layer.out_features) for layer in custom.network.networks[0]
              if isinstance(layer, nn.Linear)]
    require(shapes == [(3, 7), (7, 5), (5, 1)], "Explicit architecture settings were ignored.")
    return {"passed": True, "defaults": rows, "generic_stocks": 2,
            "generic_trading_instruments": 3, "both_methods_use_all_generic_outputs": True}


def check_generic_accounting():
    settings = tiny_settings(number_of_days=2,
        stocks=(StockSettings("Alpha"), StockSettings("Beta", initial_price=50, strike=50)),
        instruments=(InstrumentSettings("Alpha", multiplier=1, cost_rate=.001),
                     InstrumentSettings("Beta", stock_index=1, multiplier=10, cost_rate=.002),
                     InstrumentSettings("Gamma", kind="observed", multiplier=2, cost_rate=.003)))
    prices = np.array([[[100., 50., 20.], [105., 48., 22.], [102., 52., 21.]]])
    data = DataSet(prices, prices[:, :, :2], np.full((1, 3, 2), .04), np.array([7.25]),
                   cashflows=np.array([[[1., 0., .2], [0., .5, 0.]]]),
                   metadata={"kind": "software_test_fixture", "source": "Manufactured hand-calculation fixture"})
    holdings = torch.tensor([[[.5, -.2, 1.5], [.25, .3, -.5]]], dtype=torch.float64)
    errors = []
    for method in ("ours", "paper"):
        experiment = generalNN(method, settings)
        actual_loss, actual_cost = experiment.calculate_losses(data, holdings)
        ledger_loss, ledger_cost = cash_ledger(data, holdings.numpy(), experiment.settings)
        require(np.allclose(actual_loss.numpy(), [-18.89975], rtol=0, atol=1e-12), "The three-asset hand calculation failed.")
        require(np.allclose(actual_cost.numpy(), [1.20025], rtol=0, atol=1e-12), "Instrument fees or multipliers are incorrect.")
        error = max(float(np.max(np.abs(actual_loss.numpy() - ledger_loss))),
                    float(np.max(np.abs(actual_cost.numpy() - ledger_cost))))
        require(error < 1e-11, "Vector accounting and the cash ledger disagree.")
        errors.append(error)
    return {"passed": True, "fixture_is_manufactured_test_data": True,
            "expected_loss": -18.89975, "expected_cost": 1.20025,
            "maximum_cash_ledger_error": max(errors),
            "per_instrument_multipliers_costs_and_cashflows_checked": True}


def train_smoke(method, dynamic_data):
    settings = tiny_settings(instruments=(InstrumentSettings("Stock"),
                        InstrumentSettings("Stock variance", kind="variance", multiplier=1000.)))
    experiment = generalNN(method, settings, dynamic_data=dynamic_data)
    if not dynamic_data:
        supplied = experiment._generate(7, np.random.default_rng(93871), 93871, "training")
        experiment.load_data(supplied, "training")
    experiment.prepare_data()
    held_out = {name: data_rows(experiment.data[name]) for name in ("validation", "test")}
    require(not held_out["validation"] & held_out["test"], "Validation and test rows overlap.")
    data_before = {name: experiment.data[name].fingerprint() for name in ("validation", "test")}
    observed_training = []

    def inspect_forward(module, inputs):
        if not hasattr(module, "networks") or len(inputs) != 2:
            return
        stock, variance = inputs
        rows = {hashlib.sha256(stock[index].detach().numpy().tobytes()
                              + variance[index].detach().numpy().tobytes()).hexdigest()
                for index in range(len(stock))}
        require(not rows & held_out["test"], "Training forwarded held-out test states.")
        if module.training:
            require(not rows & held_out["validation"], "Validation states entered training mode.")
            observed_training.append(len(stock))

    hook = nn.modules.module.register_module_forward_pre_hook(inspect_forward)
    try:
        experiment.train(verbose=False)
    finally:
        hook.remove()
    require(len(observed_training) == 5 and len(experiment.batch_records) == 5,
            "The tiny fit did not perform the expected five updates.")
    require(all(experiment.data[name].fingerprint() == fingerprint for name, fingerprint in data_before.items()),
            "Training changed held-out data.")
    for stage, batches in experiment.training_batches.items():
        records = [record for record in experiment.batch_records if record["stage"] == stage]
        require(len(records) == len(batches), "Retained batch histories are incomplete.")
        if dynamic_data and len(batches) > 1:
            require(len({batch.fingerprint() for batch in batches}) == len(batches),
                    "Successive dynamic batches within a stage are not fresh.")
        for record, batch in zip(records, batches):
            require(len(record["losses"]) == settings.batch_size and len(record["costs"]) == settings.batch_size,
                    "Per-path batch outcomes were not retained.")
            if not dynamic_data:
                indices = np.asarray(record["pool_indices"])
                pool = experiment.data["training"]
                require(len(np.unique(indices)) < len(indices), "Replacement sampling was not exercised.")
                for field in ("prices", "stock_prices", "variances", "payoffs", "cashflows"):
                    require(torch.equal(getattr(batch, field), getattr(pool, field)[indices]),
                            "Static sampling changed or misaligned a dataset field.")
    for report in experiment.runs:
        require(report["best_validation_risk"] == min(point["risk"] for point in report["history"]),
                "Checkpoint selection did not follow validation risk.")

    weights_before = canonical({name: model.state_dict() for name, model in experiment.models.items()})
    measurements = experiment.validate()
    require(len(measurements) == 2 and experiment.measurements, "validate() did not store both policy measurements.")
    require(weights_before == canonical({name: model.state_dict() for name, model in experiment.models.items()}),
            "Validation changed weights or normalization buffers.")
    for report in experiment.runs:
        metric = "cvar" if report["objective"] == "CVaR" else "entropic"
        require(np.isclose(measurements[report["run_id"]][metric], report["best_validation_risk"],
                           rtol=1e-10, atol=1e-10), "Validation does not reproduce the selected checkpoint score.")
    experiment.evaluate()
    return experiment, {"passed": True, "method": method, "dynamic_data": dynamic_data,
                        "traded_instruments": experiment.settings.output_size,
                        "optimizer_updates_observed": 5, "test_paths_used_in_training": False}


def historical_fixture(experiment):
    """Invented prices/payments to test the API, not historical observations."""
    stock = np.array([[100., 102., 99., 104.], [100., 98., 101., 100.]])
    claim = np.array([[.003, .004, .002, .001], [.003, .002, .003, .004]])
    prices = np.stack([stock, claim], axis=2)
    dates = np.array([["2020-01-06T16:00", "2020-01-07T16:00", "2020-01-08T16:00", "2020-01-09T16:00"],
                      ["2020-02-03T16:00", "2020-02-04T16:00", "2020-02-05T16:00", "2020-02-06T16:00"]],
                     dtype="datetime64[ns]")
    flows = np.zeros((2, 3, 2))
    flows[:, 1, 0] = [.25, .10]
    return {"prices": prices, "stock_prices": stock[:, :, None],
            "variances": np.full((2, 4, 1), .04), "payoffs": np.array([8.5, 1.75]),
            "cashflows": flows, "dates": dates,
            "feature_times": dates[:, :-1] - np.timedelta64(1, "m"),
            "instrument_names": [item.name for item in experiment.settings.instruments],
            "stock_names": [item.name for item in experiment.settings.stocks],
            "metadata": {"kind": "historical", "source": "SYNTHETIC UNIT-TEST FIXTURE; NOT REAL MARKET DATA",
                         "payoff_source": "Invented terminal payments for an accounting test only",
                         "variance_source": "Fixed test numbers, available before each test execution",
                         "price_basis": "raw_per_unit", "interest_rate": 0.0,
                         "liability_timing": "terminal", "asset_cashflows": "included"}}


def check_historical(experiment):
    fixture = historical_fixture(experiment)
    invalid = copy.deepcopy(fixture)
    invalid.pop("payoffs")
    rejects(lambda: experiment.evaluate_historical(invalid), "Historical evaluation inferred a missing payoff.")
    invalid = copy.deepcopy(fixture)
    invalid["feature_times"][0, 0] = invalid["dates"][0, 0] + np.timedelta64(1, "ns")
    rejects(lambda: experiment.evaluate_historical(invalid), "A future feature timestamp was accepted.")
    invalid = copy.deepcopy(fixture)
    invalid["metadata"].pop("payoff_source")
    rejects(lambda: experiment.evaluate_historical(invalid), "Missing payoff provenance was accepted.")
    invalid = copy.deepcopy(fixture)
    invalid["metadata"]["interest_rate"] = .01
    rejects(lambda: experiment.evaluate_historical(invalid), "An unsupported interest convention was accepted.")
    invalid = copy.deepcopy(fixture)
    invalid.pop("cashflows")
    rejects(lambda: experiment.evaluate_historical(invalid), "Declared cashflows were silently replaced by zero.")
    rejects(lambda: DataSet(**invalid), "The DataSet constructor bypassed the explicit-cashflow requirement.")

    original = snapshot(experiment, False)
    with patch.object(experiment, "_generate", side_effect=AssertionError("Historical evaluation generated substitute data.")):
        view = experiment.evaluate_historical(fixture, name="Manufactured accounting fixture only")
    require(snapshot(experiment, False) == original, "Historical evaluation changed the parent experiment.")
    require(view is not experiment and view.models is experiment.models,
            "Historical evaluation did not return a separate view of frozen models.")
    require(experiment.historical_results[view.name] is view, "The historical view was not retained.")
    require(np.array_equal(view.arrays["benchmark_loss_no_hedge"], fixture["payoffs"]),
            "The supplied historical liability was replaced by a synthetic call.")
    require(not view.metadata["default_call_payoff"], "The observed liability was labeled as a default synthetic call.")
    view_before = snapshot(view)
    with patch("scripts.general_nn.simulate_paths", side_effect=AssertionError("Historical reevaluation simulated new prices.")):
        with patch.object(view, "_generate", side_effect=AssertionError("Historical reevaluation generated substitute data.")):
            with patch.object(view, "_calibrate", side_effect=AssertionError("Historical reevaluation recalibrated normalization.")):
                require(view.evaluate() is view, "Historical reevaluation did not return the existing view.")
                rejects(lambda: view.evaluate(copy.deepcopy(fixture)),
                        "A historical view accepted replacement evaluation data.")
                rejects(lambda: view.evaluate(include_delta=True),
                        "A historical view accepted the synthetic Heston delta benchmark.")
    require(snapshot(view) == view_before, "Historical reevaluation changed data, results or model state.")
    require(snapshot(experiment, False) == original, "Historical reevaluation changed the parent experiment.")
    errors = []
    for report in view.runs:
        data = view.data["historical"]
        holdings = view.predict(data, report["objective"], report["replicate_id"])
        losses, costs = cash_ledger(data, holdings.numpy(), view.settings)
        error = max(float(np.max(np.abs(view.losses(report["objective"]) - losses))),
                    float(np.max(np.abs(view.costs(report["objective"]) - costs))))
        errors.append(error)
    require(max(errors) < 1e-10, "Historical results disagree with the independent cash ledger.")
    return view, {"passed": True, "fixture_is_manufactured_not_real_data": True,
                  "maximum_independent_cash_ledger_error": max(errors),
                  "observed_payoffs_and_interval_cashflows_used": True,
                  "causal_metadata_and_missing_fields_rejected": True,
                  "historical_reevaluation_uses_existing_data_only": True,
                  "historical_replacement_data_and_synthetic_delta_rejected": True,
                  "parent_main_results_and_models_unchanged": True}


def check_roundtrip(experiment, folder):
    before = snapshot(experiment)
    experiment.save(folder)
    require([path.name for path in folder.iterdir()] == ["experiment.pt"], "Persistence did not use one snapshot file.")
    restored = generalNN.load(folder)
    after = snapshot(restored)
    changed = [name for name in before if before[name] != after[name]]
    require(not changed, f"The experiment roundtrip changed: {changed}")
    largest = 0.0
    for report in experiment.runs:
        arguments = (report["objective"], report["replicate_id"])
        expected = experiment.predict(experiment.data["test"], *arguments)
        actual = restored.predict(restored.data["test"], *arguments)
        largest = max(largest, float(torch.max(torch.abs(expected - actual))))
    require(largest == 0.0, "The loaded policies changed their predictions.")
    return {"passed": True, "prediction_maximum_error": largest,
            "historical_views_roundtripped": len(experiment.historical_results),
            "states_data_histories_measurements_and_results_preserved": True}


def check_saved_results():
    folder = PROJECT_ROOT / "results"
    summary = json.loads((folder / "summary.json").read_text())
    original = {row["run_id"]: row for row in summary["replicates"]}
    comparisons = []
    with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
        for family in ("ours", "paper_full", "paper_stock"):
            with patch("scripts.general_nn.simulate_paths", side_effect=AssertionError("Archive loading simulated paths.")):
                experiment = generalNN.from_saved(family)
            old_prices = archive["example_prices"]
            variances = archive["example_variances"][:, :, None]
            prices = old_prices[:, :, :1] if family == "ours" else old_prices
            data = DataSet(prices, old_prices[:, :, :1], variances,
                           np.maximum(old_prices[:, -1, 0] - experiment.settings.strike, 0))
            for report in experiment.runs:
                run_id = report["run_id"]
                require(report["metrics"] == original[run_id]["metrics"], "A preserved reported score changed.")
                losses = archive[report["loss_key"]]
                costs = archive["costs_" + run_id]
                require(np.array_equal(experiment.losses(report["objective"], report["replicate_id"]), losses),
                        "A preserved loss array changed.")
                require(np.array_equal(experiment.costs(report["objective"], report["replicate_id"]), costs),
                        "A preserved cost array changed.")
                measured = independent_metrics(losses, costs, experiment.settings)
                require(all(np.isclose(value, report["metrics"][name], atol=1e-10, rtol=1e-10)
                            for name, value in measured.items()), "Saved risk summaries do not reproduce from their arrays.")
                path = folder / "runs" / f"{family}_{report['replicate_id']}" / f"{report['objective']}.pt"
                state = torch.load(path, map_location="cpu", weights_only=True)
                loaded = experiment.models[run_id].state_dict()
                require(state.keys() == loaded.keys() and all(torch.equal(value, loaded[name]) for name, value in state.items()),
                        "A saved checkpoint was partially loaded or altered.")
                holdings = experiment.predict(data, report["objective"], report["replicate_id"])
                replayed_loss, replayed_cost = experiment.calculate_losses(data, holdings)
                loss_error = float(np.max(np.abs(replayed_loss.numpy() - losses[:len(data)])))
                cost_error = float(np.max(np.abs(replayed_cost.numpy() - costs[:len(data)])))
                require(max(loss_error, cost_error) < 1e-11, "Legacy example-path accounting changed.")
                comparisons.append({"run_id": run_id, "example_loss_error": loss_error,
                                    "example_cost_error": cost_error})
    require(len(comparisons) == 18, "Not all eighteen checkpoints were verified.")
    return {"passed": True, "checkpoints_checked": 18, "saved_scores_and_arrays_unchanged": True,
            "comparisons": comparisons}


def check_plot_api(ours, paper, historical):
    from scripts import plots
    checks = []
    before = [snapshot(item) for item in (ours, paper, historical)]
    for name in ("plot_paths", "plot_learning", "plot_losses", "plot_risk", "plot_loss_tails"):
        for label, value in (("ours_multi", ours), ("paper", paper),
                             ("matched", [paper, ours]), ("historical_test_fixture", historical)):
            shown = []
            texts = []
            def capture():
                figure = plt.gcf()
                shown.append(len(figure.axes))
                texts.extend(item.get_text() for item in figure.texts)
                texts.extend(axis.get_xlabel() + " " + axis.get_title() for axis in figure.axes)
            with patch("matplotlib.pyplot.show", side_effect=capture):
                with patch("scripts.general_nn.simulate_paths", side_effect=AssertionError("Plotting simulated paths.")):
                    with patch.object(generalNN, "train", side_effect=AssertionError("Plotting trained a model.")):
                        result = getattr(plots, name)(value)
            require(result is None and len(shown) == 1 and shown[0] > 0, "A plot helper did not display one figure.")
            if label == "historical_test_fixture" and name == "plot_paths":
                wording = " ".join(texts).lower()
                require("observed" in wording or "historical" in wording,
                        "Historical path graphics were not identified as observed data.")
                require("possible future" not in wording and "(s_t-k)" not in wording,
                        "A historical plot invented the synthetic call problem.")
            checks.append({"helper": name, "input": label, "axes": shown[0]})
            plt.close("all")
    require(before == [snapshot(item) for item in (ours, paper, historical)], "Plotting mutated an experiment.")
    return {"passed": True, "checks": checks, "training_or_simulation": False,
            "historical_graph_fixture_is_test_only": True}


ORACLE_PROGRAM = r'''
import hashlib, json, sys
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
project, output, version = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(project))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
if version == "old":
    from scripts.experiment import HedgingExperiment as Experiment
    from scripts.config import Settings
    from scripts.market import simulate_paths
else:
    from scripts.general_nn import generalNN as Experiment, simulate_paths
    from scripts.settings import Settings
from scripts import plots
objects = [Experiment.from_saved(family) for family in ("paper_full", "paper_stock", "ours")]
output.mkdir(parents=True, exist_ok=True)
for name in ("plot_paths", "plot_learning", "plot_losses", "plot_risk", "plot_loss_tails"):
    with patch("matplotlib.pyplot.show", side_effect=lambda name=name: plt.gcf().savefig(output/(name+".png"), dpi=100)):
        getattr(plots, name)(objects)
for number in (2, 11):
    with patch("matplotlib.pyplot.show", side_effect=lambda number=number: plt.gcf().savefig(output/("published_figure_"+str(number)+".png"), dpi=100)):
        plots.show_published_figure(number)
settings = Settings(number_of_days=3, simulation_substeps=8)
prices, variances = simulate_paths(12, np.random.default_rng(9321), settings, use_numba=False)
np.savez(output/"small_market.npz", prices=prices, variances=variances)
'''


def check_oracle(oracle, temporary):
    oracle = Path(oracle).resolve()
    require(oracle != PROJECT_ROOT.resolve(), "The immutable oracle must be the previous project.")
    financial_paths = [Path("results/summary.json"), Path("results/arrays.npz")]
    financial_paths.extend(path.relative_to(oracle) for path in sorted((oracle / "results/runs").rglob("*")) if path.is_file())
    original_hashes = {str(name): hashlib.sha256((oracle / name).read_bytes()).hexdigest() for name in financial_paths}
    for name, expected in original_hashes.items():
        require(hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() == expected,
                f"A preserved financial archive file changed: {name}")
    for label, project in (("old", oracle), ("new", PROJECT_ROOT)):
        subprocess.run([sys.executable, "-c", ORACLE_PROGRAM, str(project), str(temporary / label), label],
                       check=True, capture_output=True, text=True)
    images = []
    for old_path in sorted((temporary / "old").glob("*.png")):
        old_pixels = plt.imread(old_path)
        new_pixels = plt.imread(temporary / "new" / old_path.name)
        require(np.array_equal(old_pixels, new_pixels), f"Figure pixels changed: {old_path.name}")
        images.append({"figure": old_path.name, "shape": list(new_pixels.shape),
                       "pixel_sha256": hashlib.sha256(new_pixels.tobytes()).hexdigest()})
    require(len(images) == 7, "Not all seven baseline figures were compared.")
    with np.load(temporary / "old/small_market.npz") as old, np.load(temporary / "new/small_market.npz") as new:
        market_errors = {name: float(np.max(np.abs(old[name] - new[name]))) for name in old.files}
    require(max(market_errors.values()) == 0.0, "The extracted default simulator changed its path values.")
    for name, expected in original_hashes.items():
        require(hashlib.sha256((oracle / name).read_bytes()).hexdigest() == expected, "The old oracle was modified.")
    return {"passed": True, "preserved_financial_files": len(financial_paths),
            "figures_pixel_identical": images, "small_market_maximum_errors": market_errors,
            "oracle_financial_files_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="Optional immutable previous-project folder.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/refactor_check.json")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "historical_fixtures_are_manufactured_software_tests_only": True, "checks": {}}
    report["checks"]["defaults_and_generic_outputs"] = check_defaults_and_generic_policy()
    report["checks"]["generic_accounting"] = check_generic_accounting()
    ours, dynamic = train_smoke("ours", True)
    paper, static = train_smoke("paper", False)
    report["checks"]["dynamic_training"] = dynamic
    report["checks"]["static_training"] = static
    historical, report["checks"]["historical_evaluation"] = check_historical(ours)
    with tempfile.TemporaryDirectory(prefix="generalnn_refactor_check_") as folder:
        folder = Path(folder)
        report["checks"]["dynamic_roundtrip"] = check_roundtrip(ours, folder / "dynamic")
        report["checks"]["static_roundtrip"] = check_roundtrip(paper, folder / "static")
        report["checks"]["saved_results"] = check_saved_results()
        report["checks"]["plots"] = check_plot_api(ours, paper, historical)
        report["checks"]["oracle_preservation"] = (check_oracle(arguments.oracle, folder / "oracle")
                                                     if arguments.oracle else {"status": "not_requested"})
    report["status"] = "passed"
    report["scope"] = "Two tiny five-update fits; no full retraining or new full-sized financial experiment."
    report["source_sha256"] = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for directory in ("scripts", "checks")
                               for path in sorted((PROJECT_ROOT / directory).glob("*.py"))}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(arguments.output),
                      "check_groups": list(report["checks"])}, indent=2))


if __name__ == "__main__":
    main()
