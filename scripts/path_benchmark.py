"""Create equal numbers of market scenarios in fresh benchmark processes."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_path_benchmark(model, total_paths, output_dir, *, interval=0.01):
    """Benchmark a fresh synthetic configuration without training a policy.

Fixed mode creates and retains one dataset. Fresh mode consumes batches,
including configured Fast prefetch, and discards them after inspection.
The same total number of scenarios is created in both modes.
"""
    backend = getattr(model, "data_generator", None)
    if (not isinstance(backend, str) or backend not in {"Regular", "Fast"}
            or model.data or model.runs
            or model.metadata.get("custom_generator_requires_reconnection", False)):
        raise ValueError("Supply a fresh model using Regular or Fast generation.")
    if type(total_paths) is not int or total_paths < 1:
        raise ValueError("total_paths must be a positive integer.")
    if total_paths % model.settings.batch_size:
        raise ValueError("total_paths must be divisible by batch_size.")
    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be positive and finite.")

    root = Path(__file__).resolve().parents[1]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mode = "fresh" if model.dynamic_data else "fixed"
    stem = f"{mode}_{backend.lower()}"
    log_path = destination / f"{stem}.log"
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(model.settings.torch_threads)

    with tempfile.TemporaryDirectory(prefix="hedging_paths_") as directory:
        config_path = Path(directory) / "configuration.json"
        report_path = Path(directory) / "report.json"
        configuration = dict(
            settings=model.settings.to_dict(), method=model.method, name=model.name,
            dynamic_data=model.dynamic_data, data_generator=backend,
            fast_workers=model.fast_workers, fast_prefetch=model.fast_prefetch,
            total_paths=total_paths, interval=interval,
        )
        config_path.write_text(json.dumps(configuration), encoding="utf-8")
        print(f"Path creation: {model.name}; {total_paths:,} scenarios", flush=True)
        worker = subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.path_benchmark",
             "--config", str(config_path), "--report", str(report_path)],
            cwd=root, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for line in worker.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
            if worker.wait():
                raise RuntimeError(f"Path benchmark failed; see {log_path}.")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            (destination / f"{stem}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            return report
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            worker.stdout.close()


def _worker(config_path, report_path):
    import gc
    import hashlib
    import platform
    import time

    import numpy as np
    import torch

    from scripts.general_nn import generalNN
    from scripts.memory_tracker import track_memory
    from scripts.settings import Settings

    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = Settings.from_dict(configuration["settings"])
    torch.set_num_threads(settings.torch_threads)
    model = generalNN(
        configuration["method"], settings=settings,
        dynamic_data=configuration["dynamic_data"], name=configuration["name"],
        data_generator=configuration["data_generator"],
        fast_workers=configuration["fast_workers"],
        fast_prefetch=configuration["fast_prefetch"],
    )

    def array_bytes(data):
        seen = set()
        total = 0
        for name in ("prices", "stock_prices", "variances", "payoffs", "cashflows"):
            storage = getattr(data, name).untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    try:
        # An unrelated RNG warms compilation and workers before measurement.
        start = time.perf_counter()
        warmup = model._generate(16, np.random.default_rng(1234), 1234, "path_warmup")
        del warmup
        warmup_seconds = time.perf_counter() - start
        gc.collect()

        seed = settings.training_seed
        generator = np.random.default_rng(seed)
        total_paths = configuration["total_paths"]
        generated = 0
        maximum_dataset_bytes = 0
        checksum = 0.0
        with track_memory(model.name, configuration["interval"]) as memory:
            start_cpu = time.process_time()
            if model.dynamic_data:
                updates = total_paths // settings.batch_size
                with model._training_batch_stream(generator, seed, updates) as batches:
                    for _ in range(updates):
                        batch, _ = next(batches)
                        generated += len(batch)
                        maximum_dataset_bytes = max(maximum_dataset_bytes, array_bytes(batch))
                        checksum += batch.stock_prices[:, -1].sum().item()
                        del batch
            else:
                batch = model._generate(total_paths, generator, seed, "training")
                generated = len(batch)
                maximum_dataset_bytes = array_bytes(batch)
                checksum = batch.stock_prices[:, -1].sum().item()
                # Keep the full dataset alive through the end of measurement.
            cpu_seconds = time.process_time() - start_cpu
        if generated != total_paths:
            raise RuntimeError("The benchmark generated the wrong number of scenarios.")
        common = settings.to_dict()
        common.pop("method", None)
        root = Path(__file__).resolve().parents[1]
        files = ("scripts/general_nn.py", "scripts/settings.py", "scripts/fast_paths.py",
                 "scripts/memory_tracker.py", "scripts/path_benchmark.py")
        report = dict(
            name=model.name, dynamic_data=model.dynamic_data,
            data_generator=model.data_generator, paths_generated=generated,
            batch_size=settings.batch_size, fast_workers=model.fast_workers,
            fast_prefetch=model.fast_prefetch, memory=memory, cpu_seconds=cpu_seconds,
            warmup_seconds=warmup_seconds, maximum_dataset_bytes=maximum_dataset_bytes,
            checksum=checksum, settings=settings.to_dict(),
            comparison_signature=hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest(),
            source_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
            versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
            scope="CPU market-scenario creation, quotes, payoff, DataSet checks and cashflows; "
                  "includes batch inspection/prefetch; excludes network training, startup and warmup.",
        )
        Path(report_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        model.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()
    _worker(arguments.config, arguments.report)
