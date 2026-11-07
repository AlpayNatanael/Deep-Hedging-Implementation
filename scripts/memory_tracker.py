"""Measure notebook RAM or run a synthetic experiment in a fresh process.

Install once in the notebook: %pip install psutil

Usage:
    from scripts.memory_tracker import track_memory

    with track_memory(model.name) as memory:
        model.train(verbose=True)
        model.evaluate()

    print(memory)

Measures the current Python process's resident RAM, including CPU tensors and
existing notebook objects. GPU memory and other worker processes are excluded.
The peak is sampled, so spikes shorter than the interval can be missed.
For comparable total peaks, run each model in a freshly restarted kernel.
run_memory_experiment() handles process isolation automatically and returns
small reports; it leaves the notebook's configuration object untrained.
"""

from contextlib import contextmanager


@contextmanager
def track_memory(label="Experiment", interval=0.01):
    """Measure CPU RAM while the code inside the with block executes.

    Yield a dictionary, filled when the block finishes, with:
      baseline_mib: RAM already occupied immediately before the block.
      peak_mib: largest sampled TOTAL process RAM during the block.
      peak_extra_mib: peak_mib minus baseline_mib, bounded below by zero.
      seconds: elapsed time inside the block, including data preparation.

    peak_extra_mib is an observed increase over the starting footprint; it
    does not identify which model owns the memory. Restart between models
    for a clean comparison. MiB means 1024**2 bytes.
    """
    import json
    import math
    import os
    import subprocess
    import sys
    import time

    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a positive, finite number of seconds.")

    # A separate process can keep sampling while NumPy, Numba, or PyTorch
    # is busy. Only the notebook process's RAM is counted, not the monitor.
    monitor_code = r'''
import json, sys, threading
import psutil

process = psutil.Process(int(sys.argv[1]))
interval = float(sys.argv[2])
baseline = peak = process.memory_info().rss
stop = threading.Event()

# Stop when the notebook sends a byte, or when its input pipe closes.
def wait_for_stop():
    sys.stdin.read(1)
    stop.set()

threading.Thread(target=wait_for_stop, daemon=True).start()
print(json.dumps({"baseline": baseline}), flush=True)
try:
    while not stop.wait(interval):
        peak = max(peak, process.memory_info().rss)
    peak = max(peak, process.memory_info().rss)
except psutil.NoSuchProcess:
    pass
print(json.dumps({"peak": peak}), flush=True)
'''
    monitor = subprocess.Popen(
        [sys.executable, "-u", "-c", monitor_code, str(os.getpid()), str(interval)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    memory = {}
    try:
        # Wait for the monitor to be ready before starting the experiment.
        ready = monitor.stdout.readline()
        if not ready:
            _, error = monitor.communicate()
            raise RuntimeError("Memory monitor could not start. Install psutil. " + error.strip())
        baseline = json.loads(ready)["baseline"]
        start = time.perf_counter()
        try:
            yield memory
        finally:
            # Finish measurement even when training raises an exception.
            elapsed = time.perf_counter() - start
            output, error = monitor.communicate(input="s", timeout=10)
            if monitor.returncode:
                raise RuntimeError("Memory monitor failed: " + error.strip())
            peak = json.loads(output)["peak"]
            memory.update(
                baseline_mib=baseline / 1024**2,
                peak_mib=peak / 1024**2,
                peak_extra_mib=max(0, peak - baseline) / 1024**2,
                seconds=elapsed,
            )
            print(
                f"{label}: baseline {memory['baseline_mib']:.1f} MiB | "
                f"sampled peak {memory['peak_mib']:.1f} MiB | "
                f"peak increase {memory['peak_extra_mib']:.1f} MiB | "
                f"{elapsed:.1f} s"
            )
    finally:
        # A failed measurement must not leave its monitor running.
        if monitor.poll() is None:
            monitor.kill()
            monitor.communicate()


def run_memory_experiment(model, output_dir="results/portfolio_memory_comparison", *,
                          interval=0.01, verbose=True):
    """Train and evaluate a fresh copy of a synthetic configuration in isolation.

    Only the supplied model's settings, method, data mode and name are used.
    Its weights and data are not transferred or changed. The fresh process
    uses the existing generalNN training, simulator and evaluation methods.
    Return a JSON-compatible report containing memory, timing and test risks.
    Reports and logs are saved, without storing the large training dataset.
    """
    import json
    import math
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be positive and finite.")
    backend = getattr(model, "data_generator", None)
    if backend is None:
        backend = "Regular"  # Also accept models created by the original class.
    if (not isinstance(backend, str) or backend not in {"Regular", "Fast"}
            or model.data or model.runs
            or model.metadata.get("custom_generator_requires_reconnection", False)):
        raise ValueError("Supply a fresh model using the built-in Regular or Fast generator.")

    # The worker imports the same scripts as the notebook, including on Windows.
    project_root = Path(__file__).resolve().parents[1]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mode = "fresh" if model.dynamic_data else "fixed"
    stem = f"{model.method}_{mode}" + ("_fast" if backend == "Fast" else "")
    final_report = destination / f"{stem}.json"
    log_path = destination / f"{stem}.log"
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[variable] = str(model.settings.torch_threads)

    # Temporary configuration files never become part of the experiment data.
    with tempfile.TemporaryDirectory(prefix="hedging_memory_") as directory:
        config_path = Path(directory) / "configuration.json"
        report_path = Path(directory) / "report.json"
        config_path.write_text(json.dumps({
            "settings": model.settings.to_dict(), "method": model.method,
            "dynamic_data": model.dynamic_data, "name": model.name,
            "data_generator": backend,
            "fast_workers": getattr(model, "fast_workers", 6),
            "fast_prefetch": getattr(model, "fast_prefetch", 2),
            "interval": interval, "verbose": bool(verbose),
        }), encoding="utf-8")
        command = [sys.executable, "-u", "-m", "scripts.memory_tracker",
                   "--config", str(config_path), "--report", str(report_path)]
        print(f"Running {model.name} in a fresh process...", flush=True)
        worker = subprocess.Popen(
            command, cwd=project_root, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        try:
            # Forward progress to the notebook and retain it for inspection.
            with log_path.open("w", encoding="utf-8") as log:
                for line in worker.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            if worker.wait():
                raise RuntimeError(f"Memory experiment failed; see {log_path}.")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            final_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
            return report
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            if worker.stdout is not None:
                worker.stdout.close()


def _run_worker(config_path, report_path):
    """Worker entry point; never called directly by the notebook."""
    import gc
    import hashlib
    import json
    from pathlib import Path
    import platform
    import sys

    import numpy as np
    import psutil
    import torch
    from scripts.general_nn import generalNN, SIMULATION_BACKEND
    from scripts.settings import Settings

    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = Settings.from_dict(configuration["settings"])
    model = generalNN(method=configuration["method"], settings=settings,
                      dynamic_data=configuration["dynamic_data"], name=configuration["name"],
                      data_generator=configuration.get("data_generator", "Regular"),
                      fast_workers=configuration.get("fast_workers", 6),
                      fast_prefetch=configuration.get("fast_prefetch", 2))
    torch.set_num_threads(settings.torch_threads)

    # Warm the simulator using an unrelated seed. Compilation is excluded
    # from elapsed time, and none of these paths enter training or testing.
    model._generate(16, np.random.default_rng(1234), 1234, "memory_monitor_warmup")
    gc.collect()
    with track_memory(model.name, interval=configuration["interval"]) as memory:
        model.train(verbose=configuration["verbose"])
        model.evaluate()

    # Count backing tensor storage once, including zero cashflow arrays.
    # This is retained DATA memory; sampled process RSS includes other costs.
    seen = set()
    dataset_bytes = {}
    for name, data in model.data.items():
        size = 0
        for field in ("prices", "stock_prices", "variances", "payoffs", "cashflows"):
            storage = getattr(data, field).untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                size += storage.nbytes()
        dataset_bytes[name] = size

    # Ignore the preset's family label when checking that comparison settings
    # match. All numerical settings, instruments and network choices remain.
    common_settings = model.settings.to_dict()
    common_settings.pop("method", None)
    signature = hashlib.sha256(json.dumps(common_settings, sort_keys=True).encode()).hexdigest()
    project_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        path: hashlib.sha256((project_root / path).read_bytes()).hexdigest()
        for path in ("scripts/general_nn.py", "scripts/settings.py", "scripts/memory_tracker.py",
                     "scripts/fast_paths.py") if (project_root / path).exists()
    }
    risks = {}
    for objective in ("CVaR", "Entropic"):
        scores = [run["metrics"][objective.lower()] for run in model.runs
                  if run["objective"] == objective]
        risks[objective] = {"mean": float(np.mean(scores)), "runs": scores}
    report = {
        "name": model.name, "method": model.method,
        "dynamic_data": model.dynamic_data,
        "data_generator": model.data_generator,
        "fast_workers": model.fast_workers, "fast_prefetch": model.fast_prefetch,
        "memory": memory, "sampling_interval_seconds": configuration["interval"],
        "parameter_count": model.parameter_count,
        "dataset_bytes": dataset_bytes, "retained_dataset_mib": sum(dataset_bytes.values()) / 1024**2,
        "test_risk": risks, "runs": model.runs,
        "test_data_id": model.metadata["test_data_id"],
        "instrument_data_ids": model.metadata["instrument_data_ids"],
        "comparison_signature": signature, "source_hashes": source_hashes,
        "settings": model.settings.to_dict(),
        "raw_training_batches_retained": sum(len(items) for items in model.training_batches.values()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "psutil": psutil.__version__,
                     "platform": sys.platform, "simulation_backend": SIMULATION_BACKEND},
        "measurement_scope": "Sampled RSS of one isolated CPU training process during data preparation, "
                             "training and test evaluation; notebook and monitor processes excluded. "
                             "Simulator compilation is warmed before timing. No GPU memory measured.",
    }
    Path(report_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run one isolated hedging memory measurement.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()
    _run_worker(arguments.config, arguments.report)
