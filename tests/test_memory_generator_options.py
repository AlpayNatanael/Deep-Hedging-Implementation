"""Check backend forwarding through the real isolated memory worker."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.general_nn import generalNN
from scripts.memory_tracker import run_memory_experiment
from scripts.settings import Settings, NetworkSettings


class MemoryGeneratorTests(unittest.TestCase):
    def settings(self):
        return Settings(number_of_days=2,simulation_substeps=2,batch_size=4,
            normalization_size=4,validation_size=4,test_size=8,static_training_size=16,
            warmup_updates=1,continuation_updates=1,validation_interval=1,
            number_of_replicates=1,network=NetworkSettings((4,4),'tanh',False))

    def test_worker_preserves_regular_and_fast_options(self):
        for backend in ['Regular','Fast']:
            model=generalNN('ours',self.settings(),data_generator=backend,
                            fast_workers=2,fast_prefetch=1)
            with tempfile.TemporaryDirectory() as directory:
                report=run_memory_experiment(model,output_dir=directory,verbose=False)
                self.assertEqual(report['data_generator'],backend)
                self.assertEqual((report['fast_workers'],report['fast_prefetch']),(2,1))
                self.assertGreater(report['memory']['peak_mib'],0)
                self.assertEqual(len(list(Path(directory).glob('*.json'))),1)
                self.assertIn('scripts/fast_paths.py',report['source_hashes'])
            self.assertFalse(model.data)
            self.assertFalse(model.runs)

    def test_custom_generator_still_rejected(self):
        model=generalNN('ours',self.settings(),data_generator=lambda *args:None)
        with self.assertRaisesRegex(ValueError,'Regular or Fast'):
            run_memory_experiment(model)

    def test_pending_custom_reconnection_rejected(self):
        model=generalNN('ours',self.settings())
        model.metadata['custom_generator_requires_reconnection']=True
        with self.assertRaisesRegex(ValueError,'Regular or Fast'):
            run_memory_experiment(model)


if __name__=='__main__':unittest.main()
