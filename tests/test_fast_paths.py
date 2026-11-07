"""Focused regressions for optional generation; run with unittest discovery."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.general_nn import generalNN, simulate_paths
from scripts.settings import NetworkSettings, Settings, StockSettings


def small_settings(**changes):
    values = dict(
        number_of_days=3, simulation_substeps=4, batch_size=4,
        normalization_size=8, validation_size=12, test_size=12,
        static_training_size=16, warmup_updates=2, continuation_updates=3,
        validation_interval=1, number_of_replicates=1,
        network=NetworkSettings((4, 4), "tanh", True),
    )
    values.update(changes)
    return Settings(**values)


class GeneratorTests(unittest.TestCase):
    def assert_same_models(self, left, right):
        self.assertEqual(set(left.models), set(right.models))
        for run_id in left.models:
            a, b = left.models[run_id].state_dict(), right.models[run_id].state_dict()
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), (run_id, name))
        for name in left.arrays:
            if name.startswith("loss_") or name.startswith("costs_"):
                np.testing.assert_array_equal(left.arrays[name], right.arrays[name])

    def assert_no_workers(self):
        self.assertFalse([thread.name for thread in threading.enumerate()
                          if thread.name.startswith(("hedging-fast", "hedging-prefetch"))])

    def test_00_regular_does_not_import_backend(self):
        code = """
import sys
from scripts.general_nn import generalNN
from scripts.settings import Settings
x = generalNN('ours', Settings(number_of_days=2, simulation_substeps=2,
    batch_size=4, normalization_size=4, validation_size=4, test_size=4,
    warmup_updates=1, continuation_updates=1, validation_interval=1,
    number_of_replicates=1), data_generator='Regular')
x.train(objectives=('CVaR',))
assert x._fast_generator is None
assert 'scripts.fast_paths' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())

    def test_selector_defaults_and_validation(self):
        for value in [None, "Regular", " regular "]:
            self.assertEqual(generalNN("ours", small_settings(), data_generator=value).data_generator, "Regular")
        self.assertEqual(generalNN("ours", small_settings()).data_generator, "Regular")
        with self.assertRaises(ValueError):
            generalNN("ours", small_settings(), data_generator="Fats")
        with self.assertRaises(TypeError):
            generalNN("ours", small_settings(), data_generator=17)
        for option, value in [("fast_workers", 0), ("fast_prefetch", -1), ("fast_workers", True)]:
            with self.assertRaises(ValueError):
                generalNN("ours", small_settings(), **{option: value})

    def test_regular_seeded_paths(self):
        experiment = generalNN("ours", small_settings(), data_generator="Regular")
        s = experiment.settings.for_stock(0)
        prices, variances = simulate_paths(17, np.random.default_rng(551), s)
        batch = experiment._generate(17, np.random.default_rng(551), 551, "training_batch")
        np.testing.assert_array_equal(batch.stock_prices.numpy()[:, :, 0], prices[:, :, 0])
        np.testing.assert_array_equal(batch.variances.numpy()[:, :, 0], variances)

    def test_validation_and_test_match_across_backends(self):
        stocks = (StockSettings(name="A"), StockSettings(name="B", initial_price=50., strike=50.))
        settings = small_settings(stocks=stocks)
        for dynamic in [True, False]:
            a = generalNN("paper", settings, dynamic_data=dynamic, data_generator="Regular")
            b = generalNN("paper", settings, dynamic_data=dynamic, data_generator="Fast")
            try:
                a.prepare_data(); b.prepare_data()
                for split in ["validation", "test"]:
                    self.assertEqual(a.data[split].fingerprint(), b.data[split].fingerprint())
            finally:
                a.close(); b.close()

    def test_fast_matches_original_kernel_for_matching_child_streams(self):
        from scripts.fast_paths import FastPathGenerator
        settings = small_settings().resolve("ours")
        simulator = FastPathGenerator(settings, workers=2)
        try:
            for count in [1, 17, 1025]:
                root_rng = np.random.default_rng(901)
                root = int(root_rng.integers(0, 2**63 - 1))
                children = np.random.SeedSequence(root).spawn(2)
                expected = [simulate_paths(size, np.random.default_rng(child), settings.for_stock(0))
                            for size, child in zip(((count + 1)//2, count//2), children) if size]
                actual = simulator.generate(count, np.random.default_rng(901))
                np.testing.assert_array_equal(actual["stock_prices"][:, :, 0],
                    np.concatenate([prices[:, :, 0] for prices, _ in expected]))
                np.testing.assert_array_equal(actual["variances"][:, :, 0],
                    np.concatenate([variances for _, variances in expected]))
        finally:
            simulator.close()

    def test_fast_deterministic_across_worker_counts(self):
        from scripts.fast_paths import FastPathGenerator
        settings = small_settings(stocks=(StockSettings(name="A"), StockSettings(name="B"))).resolve("paper")
        a, b = FastPathGenerator(settings, 1), FastPathGenerator(settings, 4)
        try:
            x, y = a.generate(17, np.random.default_rng(818)), b.generate(17, np.random.default_rng(818))
            for name in ["prices", "stock_prices", "variances", "payoffs"]:
                np.testing.assert_array_equal(x[name], y[name])
        finally:
            a.close(); b.close()

    def test_prefetch_preserves_full_training(self):
        settings = small_settings()
        a = generalNN("paper", settings, data_generator="Fast", fast_prefetch=0)
        b = generalNN("paper", settings, data_generator="Fast", fast_prefetch=2)
        a.train(); a.evaluate()
        b.train(); b.evaluate()
        self.assert_same_models(a, b)
        for split in a.data:
            self.assertEqual(a.data[split].fingerprint(), b.data[split].fingerprint())
        self.assert_no_workers()

    def test_prefetch_order_exception_and_early_exit(self):
        from scripts.fast_paths import PrefetchedBatches
        counter = iter(range(10))
        with PrefetchedBatches(lambda: next(counter), 10, 2) as batches:
            self.assertEqual(list(batches), list(range(10)))
        self.assertFalse(batches.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            def fail():
                raise RuntimeError("producer failed")
            with PrefetchedBatches(fail, 3, 2) as failed:
                next(failed)
        self.assertFalse(failed.thread.is_alive())
        with PrefetchedBatches(lambda: 1, 1000, 2) as early:
            next(early)
        self.assertFalse(early.thread.is_alive())
        self.assertTrue(early.queue.empty())

    def test_prefetch_rejects_unbounded_queue(self):
        from scripts.fast_paths import PrefetchedBatches
        with self.assertRaises(ValueError):
            PrefetchedBatches(lambda: 1, 4, ahead=0)

    def test_training_failure_closes_all_workers(self):
        experiment = generalNN("ours", small_settings(), data_generator="Fast")
        with patch("scripts.general_nn.training_objective", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                experiment.train(objectives=("CVaR",))
        self.assertIsNone(experiment._fast_generator)
        self.assert_no_workers()

    def test_saved_selector_and_old_snapshot(self):
        for mode in ["Regular", "Fast"]:
            experiment = generalNN("ours", small_settings(), data_generator=mode,
                                   fast_workers=2, fast_prefetch=1)
            self.assertEqual(experiment._snapshot()["format_version"], 2 if mode == "Fast" else 1)
            with tempfile.TemporaryDirectory() as folder:
                experiment.save(folder)
                loaded = generalNN.load(folder)
            self.assertEqual(loaded.data_generator, mode)
            self.assertEqual((loaded.fast_workers, loaded.fast_prefetch), (2, 1))
            self.assertIsNone(loaded._fast_generator)
            self.assertFalse(loaded.metadata.get("custom_generator_requires_reconnection", False))
        # A version-1 file has no selector/configuration fields.
        old = generalNN("ours", small_settings())._snapshot()
        old["format_version"] = 1
        for field in ["data_generator", "fast_workers", "fast_prefetch"]:
            old.pop(field)
        loaded = generalNN._restore(old)
        self.assertEqual(loaded.data_generator, "Regular")

    def test_custom_callback_roundtrip(self):
        template = generalNN("ours", small_settings())
        def callback(count, generator, settings):
            return template._generate(count, generator, 5, "training")
        experiment = generalNN("ours", small_settings(), data_generator=callback)
        batch = experiment._generate(5, np.random.default_rng(3), 3, "validation")
        self.assertEqual(batch.role, "validation")
        loaded = generalNN._restore(experiment._snapshot())
        with self.assertRaisesRegex(ValueError, "Reconnect"):
            loaded._generate(5, np.random.default_rng(3), 3, "validation")
        loaded.data_generator = callback
        self.assertEqual(len(loaded._generate(5, np.random.default_rng(3), 3, "validation")), 5)


if __name__ == "__main__":
    unittest.main()
