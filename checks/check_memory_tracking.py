"""Check native RAM measurement, cleanup, isolation and matched test data."""
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.general_nn import generalNN
from scripts.memory_tracker import track_memory, run_memory_experiment
from scripts.settings import Settings, StockSettings, NetworkSettings


def main():
    process = psutil.Process()
    children_before = {child.pid for child in process.children()}
    with track_memory("64 MiB native allocation") as native:
        array = np.ones(8 * 1024 * 1024, dtype=np.float64)
        time.sleep(0.1)
    assert native['peak_extra_mib'] >= 50, native
    del array
    try:
        with track_memory("Intentional exception"):
            raise ValueError('expected')
    except ValueError as error:
        assert str(error) == 'expected'
    else:
        raise AssertionError('Training exception was swallowed.')

    stocks = tuple(StockSettings(name=f'Stock{i+1}', initial_price=100.0,
                                 strike=100.0) for i in range(3))
    settings = Settings(stocks=stocks, network=NetworkSettings((36, 36), 'tanh', True),
                        warmup_updates=2, continuation_updates=2,
                        number_of_replicates=1, validation_interval=1,
                        normalization_size=64, validation_size=64, test_size=128,
                        static_training_size=256)
    fixed = generalNN(method='paper', settings=settings, dynamic_data=False)
    from dataclasses import replace
    fresh = generalNN(method='ours', settings=replace(settings, instruments=fixed.settings.instruments),
                      dynamic_data=True)
    with tempfile.TemporaryDirectory() as directory:
        reports = [run_memory_experiment(model, directory, verbose=False)
                   for model in (fixed, fresh)]
        for key in ('comparison_signature', 'test_data_id', 'instrument_data_ids', 'source_hashes'):
            assert reports[0][key] == reports[1][key], key
        assert reports[0]['parameter_count'] == reports[1]['parameter_count'] == 64980
        assert reports[0]['dataset_bytes']['training'] == 256 * 4424
        assert 'training' not in reports[1]['dataset_bytes']
        assert len(reports[0]['runs']) == len(reports[1]['runs']) == 2
        assert all(report['memory']['peak_mib'] >= report['memory']['baseline_mib'] for report in reports)
        assert all(not model.runs and not model.data for model in (fixed, fresh))
        for filename in ('paper_fixed.json', 'ours_fresh.json'):
            json.loads((Path(directory) / filename).read_text())
    assert {child.pid for child in process.children()} <= children_before
    print('Passed: native RAM, exception cleanup, isolated training, matching scenarios and parent isolation.')


if __name__ == '__main__':
    main()
