"""Check grid values, compatibility and exports without training any networks.

Run from the repository root: python checks/check_results_grid.py
Optionally pass a previous plots.py to compare against its standalone plots.
"""

import importlib.util
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import plots
from scripts.settings import InstrumentSettings, Settings


class RecordedExperiment:
    """Small recorded results with repeated runs, irregular histories and ties."""

    def __init__(self, dynamic_data):
        self.dynamic_data = dynamic_data
        self.name = "Our network: stock + variance" if dynamic_data else "Paper network: stock + variance"
        self.family = "ours" if dynamic_data else "paper_full"
        self.settings = Settings(instruments=(InstrumentSettings(),), display_replicate=1)
        self.metadata = {"test_data_id": "shared-test", "instrument_data_ids": {"Stock": "shared-quotes"}}
        baseline = np.array([0, 0, 0, 1, 4, 10, 12, 15], dtype=float)
        self.arrays = {"benchmark_loss_no_hedge": baseline}
        self.benchmarks = {"no_hedge": self.metrics(baseline)}
        self.runs = []
        self.recorded_losses = {}
        for objective in ("CVaR", "Entropic"):
            for replicate in (0, 1):
                losses = np.array([-4, -1, 0, 0, 2, 2, 8, 8], dtype=float)
                losses += replicate - int(dynamic_data)
                self.recorded_losses[objective, replicate] = losses
                updates = (0, 25, 50) if replicate == 0 else (0, 50, 75)
                risks = (12, 8, 7) if objective == "CVaR" else (80, 9, 7)
                self.runs.append({
                    "objective": objective,
                    "replicate_id": replicate,
                    "history": [{"update": u, "risk": r + replicate} for u, r in zip(updates, risks)],
                    "best_update": 50,
                    "metrics": self.metrics(losses),
                })

    @staticmethod
    def metrics(losses):
        return {"cvar": np.sort(losses)[len(losses) // 2:].mean(),
                "entropic": losses.max() + np.log(np.exp(losses - losses.max()).mean())}

    def get_run(self, objective, replicate):
        return next(r for r in self.runs if (r["objective"], r["replicate_id"]) == (objective, replicate))

    def losses(self, objective, replicate):
        return self.recorded_losses[objective, replicate]


def same_plot_data(first, second):
    """Compare rendered data, scales and annotations rather than image pixels."""
    assert first.get_xscale() == second.get_xscale()
    assert first.get_yscale() == second.get_yscale()
    np.testing.assert_array_equal(first.get_xlim(), second.get_xlim())
    np.testing.assert_array_equal(first.get_ylim(), second.get_ylim())
    assert len(first.lines) == len(second.lines)
    for a, b in zip(first.lines, second.lines):
        np.testing.assert_array_equal(a.get_xdata(), b.get_xdata())
        np.testing.assert_array_equal(a.get_ydata(), b.get_ydata())
        assert (a.get_label(), a.get_drawstyle(), a.get_linestyle()) == (b.get_label(), b.get_drawstyle(), b.get_linestyle())
    assert len(first.collections) == len(second.collections)
    for a, b in zip(first.collections, second.collections):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    assert [t.get_text() for t in first.texts] == [t.get_text() for t in second.texts]
    assert first.get_xlabel() == second.get_xlabel()
    assert first.get_ylabel() == second.get_ylabel()
    assert [t.get_text() for t in first.get_legend().get_texts()] == [t.get_text() for t in second.get_legend().get_texts()]


def check(reference=plots):
    experiments = [RecordedExperiment(False), RecordedExperiment(True)]
    before = pickle.dumps(experiments)
    figure, axes = plots.plot_results_grid(experiments, show=False, dpi=80)
    assert axes.shape == (3, 2) and len(figure.axes) == 6
    assert axes[0, 1].get_yscale() == "log"
    for row, function in enumerate(("plot_learning", "plot_risk", "plot_loss_tails")):
        originals = getattr(reference, function)(experiments, show=False, dpi=80)
        for column, objective in enumerate(("CVaR", "Entropic")):
            same_plot_data(originals[objective][1], axes[row, column])
        if reference is not plots:
            current = getattr(plots, function)(experiments, show=False, dpi=80)
            for objective in originals:
                same_plot_data(originals[objective][1], current[objective][1])
                assert originals[objective][1].get_title() == current[objective][1].get_title()

    # Tied maximum losses must keep both scenarios in the displayed frequency.
    np.testing.assert_array_equal(axes[2, 0].lines[0].get_xdata(), [-3, 0, 1, 3, 9])
    np.testing.assert_array_equal(axes[2, 0].lines[0].get_ydata(), [100, 87.5, 75, 50, 25])
    assert axes[2, 0].get_yscale() == "log"
    assert axes[2, 0].lines[0].get_xdata()[-1] == 9
    for objective in (" cVaR ", "ENTROPIC"):
        _, selected = plots.plot_results_grid(experiments, objective=objective, replicate=0, threshold=None, show=False, dpi=60)
        assert selected.shape == (3, 1)
        # No threshold line; three loss curves plus the one-scenario reference.
        assert len(selected[2, 0].lines) == 4
        np.testing.assert_array_equal(selected[0, 0].lines[0].get_xdata(), [0, 25, 50])
        assert selected[2, 0].lines[0].get_xdata()[-1] == 8
        assert len(selected[1, 0].collections[0].get_offsets()) == 2
    assert pickle.dumps(experiments) == before, "Plotting changed the experiments."

    with TemporaryDirectory() as folder:
        plots.plot_results_grid(experiments, show=False, dpi=60, save_dir=folder)
        exported = list(Path(folder).glob("results_grid_*"))
        assert {p.suffix for p in exported} == {".png", ".svg"}
        assert all(p.stat().st_size > 1000 for p in exported)

    # Both notebook and plain Python display exactly one complete grid.
    with patch("IPython.get_ipython", return_value=None), patch.object(plt, "show") as shown:
        assert plots.plot_results_grid(experiments, dpi=60) is None
        shown.assert_called_once()
    from IPython.display import SVG, Markdown
    with patch("IPython.get_ipython", return_value=SimpleNamespace(kernel=True)), patch("IPython.display.display") as displayed:
        plots.plot_results_grid(experiments, dpi=60, explain=True)
        assert displayed.call_count == 2
        assert isinstance(displayed.call_args_list[0].args[0], SVG)
        assert isinstance(displayed.call_args_list[1].args[0], Markdown)

    # Incompatible datasets and invalid arguments must fail without leaking figures.
    for case in ("paths", "quotes", "objective", "length", "threshold", "history"):
        bad = [RecordedExperiment(False), RecordedExperiment(True)]
        kwargs = {"show": False, "dpi": 60}
        if case == "paths":
            bad[1].metadata["test_data_id"] = "different-test"
        elif case == "quotes":
            bad[1].metadata["instrument_data_ids"]["Stock"] = "different-quotes"
        elif case == "objective":
            kwargs["objective"] = "unknown"
        elif case == "length":
            bad[1].recorded_losses["CVaR", 1] = np.array([1, 2])
        elif case == "threshold":
            kwargs["threshold"] = float("nan")
        else:
            bad[1].runs[0]["history"] = []
        try:
            plots.plot_results_grid(bad, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid {case}.")
        assert not plt.get_fignums(), f"Figure leaked after invalid {case}."
    experiments[1].runs = [r for r in experiments[1].runs if r["objective"] == "CVaR"]
    _, common = plots.plot_results_grid(experiments, show=False, dpi=60)
    assert common.shape == (3, 1)
    assert not plt.get_fignums()
    print("Passed: grid data, standalone compatibility, tied extremes, replicates, exports, display, validation and no mutation.")


if __name__ == "__main__":
    reference = plots
    if len(sys.argv) > 1:
        spec = importlib.util.spec_from_file_location("previous_plots", sys.argv[1])
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    check(reference)
