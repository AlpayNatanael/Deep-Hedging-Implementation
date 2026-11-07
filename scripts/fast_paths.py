"""Optional parallel Heston generation and bounded training-batch prefetch.

Imported lazily: Regular experiments never import or start this backend.
The numerical kernel is the one already used in general_nn.py.
"""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import queue
import threading

import numpy as np
from numba import njit

from .general_nn import (
    PATHS_PER_CHUNK, _simulate_fast, transition_parameters, variance_claim_price,
)


@njit(nogil=True)
def _parallel_kernel(count, generator, initial_price, initial_variance,
                     days, substeps, coefficients):
    """Release the GIL around the existing compiled numerical kernel."""
    return _simulate_fast(count, generator, initial_price, initial_variance,
                          days, substeps, coefficients)


class FastPathGenerator:
    """One reusable worker pool per experiment; call generate sequentially.

Each stock has two path blocks, regardless of worker count. Every block gets
its own Generator. The caller's RNG supplies a new root seed per dataset.
"""
    def __init__(self, settings, workers=6):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer.")
        if any(item.kind not in {"stock", "variance"} for item in settings.instruments):
            raise ValueError("Fast generation supports stock and variance instruments only.")
        self.settings = settings
        self.stocks = [settings.for_stock(i) for i in range(len(settings.stocks))]
        keys = (
            "cir_degrees", "cir_scale", "noncentrality_multiplier",
            "old_variance_weight", "next_variance_weight", "stock_exponent",
            "log_mgf_constant", "log_mgf_variance_coefficient",
            "independent_variance_fraction",
        )
        self.coefficients = []
        for stock in self.stocks:
            parameters = transition_parameters(stock)
            self.coefficients.append(tuple(float(parameters[key]) for key in keys))
        self.remaining = (settings.number_of_days - np.arange(settings.number_of_days + 1)) * settings.time_step
        self.pool = ThreadPoolExecutor(
            max_workers=min(workers, 2 * len(self.stocks)),
            thread_name_prefix="hedging-fast",
        )
        self.warmed = False
        self.stop = threading.Event()

    def _block(self, index, count, child_seed):
        stock = self.stocks[index]
        generator = np.random.default_rng(child_seed)
        shape = (count, stock.number_of_days + 1)
        output = [np.empty(shape, dtype=np.float64) for _ in range(3)]
        for first in range(0, count, PATHS_PER_CHUNK):
            if self.stop.is_set():
                raise CancelledError("Fast path generation was closed.")
            last = min(first + PATHS_PER_CHUNK, count)
            arrays = _parallel_kernel(
                last - first, generator, float(stock.initial_price),
                float(stock.initial_variance), stock.number_of_days,
                stock.simulation_substeps, self.coefficients[index],
            )
            for destination, values in zip(output, arrays):
                destination[first:last] = values
        prices, variances, integrals = output
        claim = variance_claim_price(integrals, variances, self.remaining, stock)
        return prices, variances, claim

    def generate(self, count, generator):
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer.")
        if self.pool is None:
            raise RuntimeError("This fast generator has been closed.")
        if not self.warmed:
            # Compile on the calling thread without advancing the experiment RNG.
            self._block(0, 1, 0)
            self.warmed = True
        root = int(generator.integers(0, 2**63 - 1))
        children = np.random.SeedSequence(root).spawn(2 * len(self.stocks))
        sizes = ((count + 1) // 2, count // 2)
        futures = [
            [self.pool.submit(self._block, index, size, children[2 * index + block])
             for block, size in enumerate(sizes) if size]
            for index in range(len(self.stocks))
        ]
        stocks, variances, claims = [], [], []
        for group in futures:
            pieces = [future.result() for future in group]
            stocks.append(np.concatenate([piece[0] for piece in pieces], axis=0))
            variances.append(np.concatenate([piece[1] for piece in pieces], axis=0))
            claims.append(np.concatenate([piece[2] for piece in pieces], axis=0))
        payoff = np.zeros(count, dtype=np.float64)
        for prices, stock in zip(stocks, self.settings.stocks):
            payoff += stock.call_quantity * np.maximum(prices[:, -1] - stock.strike, 0)
        quotes = [stocks[item.stock_index] if item.kind == "stock" else claims[item.stock_index]
                  for item in self.settings.instruments]
        return dict(
            prices=np.stack(quotes, axis=2), stock_prices=np.stack(stocks, axis=2),
            variances=np.stack(variances, axis=2), payoffs=payoff,
            instrument_names=[item.name for item in self.settings.instruments],
            stock_names=[stock.name for stock in self.settings.stocks],
            metadata={"kind": "synthetic", "stocks_independent": True,
                      "path_generator": "Fast", "blocks_per_stock": 2},
        )

    def close(self):
        if self.pool is not None:
            self.stop.set()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None


class PrefetchedBatches:
    """Finite, ordered prefetch with a bounded queue and exception forwarding.

Only the producer calls factory, so the stage RNG is never used concurrently.
The queue, consumer and producer can hold at most ahead + 2 live batches,
in addition to simulation scratch. close() also releases queued batches.
"""
    _END = object()

    def __init__(self, factory, count, ahead=2):
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer.")
        if type(ahead) is not int or ahead < 1:
            raise ValueError("ahead must be a positive integer.")
        self.factory, self.count = factory, count
        self.queue = queue.Queue(maxsize=ahead)
        self.stop = threading.Event()
        self.thread = None
        self.finished = False

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        try:
            for _ in range(self.count):
                if self.stop.is_set() or not self._put((True, self.factory())):
                    return
        except BaseException as error:
            self._put((False, error))
        finally:
            self._put(self._END)

    def __enter__(self):
        self.thread = threading.Thread(
            target=self._produce, name="hedging-prefetch", daemon=True,
        )
        self.thread.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        item = self.queue.get()
        if item is self._END:
            self.finished = True
            raise StopIteration
        success, value = item
        if not success:
            raise value
        return value

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def __exit__(self, *_):
        self.close()
