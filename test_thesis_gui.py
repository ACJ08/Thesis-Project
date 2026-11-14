"""Preflight and UI regression checks; no model downloads or training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from thesis_gui_schema import validate_job, sha256, DATASETS, MODEL_IDS


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        for index, name in enumerate(DATASETS):
            (self.root / 'data' / name).write_text(f'song_name,artist,text,explicitness\nsong{index},artist,hello world,Safe\n')
        self.run = self.root / 'run'
        self.run.mkdir()
        (self.run / 'best_model.pt').write_bytes(b'fixture')
        (self.run / 'manifest.json').write_text(json.dumps({'architecture': 'roberta', 'pipeline_version': 'v26'}))
        self.job = dict(schema_version=1, action='label', model='roberta', run=str(self.run),
                        source='', output='', lyrics='hello world', column='')

    def test_single_song(self):
        self.assertEqual(validate_job(self.job, self.root)['model'], 'roberta')

    def test_missing_input(self):
        with self.assertRaisesRegex(ValueError, 'Paste'):
            validate_job({**self.job, 'lyrics': ''}, self.root)

    def test_wrong_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'v26 hybrid'):
            validate_job({**self.job, 'model': 'hybrid'}, self.root)

    def test_unknown_schema_and_action(self):
        for change in ({'schema_version': 9}, {'action': 'arbitrary'}, {'model': 'unknown'}):
            with self.assertRaises(ValueError):
                validate_job({**self.job, **change}, self.root)

    def test_batch_input_output(self):
        source = self.root / 'input.csv'
        source.write_text('lyrics\nhello\n')
        job = {**self.job, 'source': str(source), 'output': str(self.root / 'labels.csv')}
        self.assertEqual(validate_job(job, self.root)['source'], str(source))
        with self.assertRaisesRegex(ValueError, 'separate'):
            validate_job({**job, 'output': str(source)}, self.root)
        source.write_text('wrong_column\nhello\n')
        with self.assertRaisesRegex(ValueError, 'column'):
            validate_job(job, self.root)

    def test_training_defaults_and_split_overlap(self):
        job = dict(schema_version=1, action='train', mode='roberta', smoke=True, datasets=['', '', ''], runs={})
        result = validate_job(job, self.root)
        self.assertTrue(all(Path(path).exists() for path in result['datasets']))
        with self.assertRaisesRegex(ValueError, 'separate'):
            validate_job({**job, 'datasets': [result['datasets'][0]] * 3}, self.root)

    def test_frozen_dataset_mismatch(self):
        hashes = dict(zip(('training', 'validation', 'test'), (sha256(self.root / 'data' / n) for n in DATASETS)))
        runs = {}
        for model in MODEL_IDS:
            folder = self.root / model
            folder.mkdir()
            (folder / 'best_model.pt').write_bytes(b'fixture')
            (folder / 'vocabulary.json').write_text('{}')
            (folder / 'manifest.json').write_text(json.dumps({'architecture': model, 'pipeline_version': 'v26',
                                                             'input_sha256': hashes, 'smoke_test': False}))
            runs[model] = str(folder)
        job = dict(schema_version=1, action='train', mode='compare', smoke=False, datasets=['', '', ''], runs=runs)
        validate_job(job, self.root)
        (self.root / 'data' / DATASETS[0]).write_text('song_name,artist,text,explicitness\nx,a,changed,Safe\n')
        with self.assertRaisesRegex(ValueError, 'hashes'):
            validate_job(job, self.root)

    def test_negative_samples_and_types(self):
        job = dict(schema_version=1, action='explain', model='roberta', run=str(self.run), method='Both',
                   lyrics='', validation='default', samples=-1)
        with self.assertRaisesRegex(ValueError, 'Sample'):
            validate_job(job, self.root)
        with self.assertRaisesRegex(ValueError, 'samples'):
            validate_job({**job, 'samples': True}, self.root)

    def test_integrity_and_malformed_manifest(self):
        manifest = self.run / 'manifest.json'
        manifest.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            validate_job(self.job, self.root)
        manifest.write_text(json.dumps({'architecture': 'roberta', 'pipeline_version': 'v26', 'checkpoint_sha256': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'Integrity'):
            validate_job(self.job, self.root)

    def test_nested_checkpoint_discovery(self):
        import thesis_gui as gui
        folder = self.root / 'experiments' / 'annotator' / 'seed42' / 'checkpoint'
        folder.mkdir(parents=True)
        (folder / 'best_model.pt').write_bytes(b'fixture')
        (folder / 'manifest.json').write_text(json.dumps({'architecture': 'roberta', 'pipeline_version': 'v26'}))
        with patch.object(gui, 'ROOT', self.root):
            self.assertEqual(gui.run_choices('roberta'), [folder])
            self.assertEqual(gui.run_choices('hybrid'), [])


class WorkerPresentationTests(unittest.TestCase):
    def test_training_charts_from_saved_metrics(self):
        # Import the worker with a synthetic classifier so no ML setup runs.
        import importlib.util
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classifier = SimpleNamespace(DATASET_OVERRIDES={}, TRAIN_CSV_FILE='train', VAL_CSV_FILE='val',
                                         TEST_CSV_FILE='test', FROZEN_RUN_DIRS={}, PROJECT_DIR=root, ALL_NAMES=['roberta'])
            def train():
                folder = root / 'experiments' / f'roberta_v26_{classifier.RUN_ID}'
                folder.mkdir(parents=True)
                (folder / 'validation_metrics.json').write_text(json.dumps({'accuracy': .8, 'macro_f1': .7,
                        'weighted_f1': .75, 'confusion_matrix': [[2,0,0],[0,2,0],[0,1,1]]}))
                (folder / 'epoch_history.csv').write_text('epoch,train_loss,validation_loss,validation_accuracy,validation_macro_f1\n1,1.2,1.1,.7,.6\n2,.9,.8,.8,.7\n')
            classifier.main_v26 = train
            with patch.dict('sys.modules', {'lyric_classifier_v26_all_models': classifier}):
                spec = importlib.util.spec_from_file_location('worker_under_test', Path(__file__).with_name('thesis_gui_worker.py'))
                worker = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(worker)
                result = worker.train_or_compare({'datasets': ['', '', ''], 'mode': 'roberta', 'smoke': True, 'runs': {}})
                titles = [chart['title'] for chart in result['charts']]
                self.assertTrue(any('Loss' in title for title in titles))
                self.assertTrue(any('Validation scores' in title for title in titles))
                self.assertTrue(any('confusion' in title for title in titles))
                self.assertTrue(Path(result['output_paths'][0]).is_dir())
                chart = worker.erasure_chart('LIME', [{'Erasure %': 5, 'Comprehensiveness': .2, 'Random_mean': .1}])
                self.assertEqual(chart['series'][1]['y'], [.1])


class InterfaceTests(unittest.TestCase):
    def test_hidden_launch_waits_for_mapped_size(self):
        import thesis_gui as gui
        app = gui.Application.__new__(gui.Application)
        app.root = Mock()
        app.split = Mock()
        app._split_initialized = False
        app.split.winfo_height.return_value = 1
        app.initialize_split()
        app.root.after_idle.assert_not_called()
        self.assertFalse(app._split_initialized)
        app.split.winfo_height.return_value = 800
        app.initialize_split()
        app.root.after_idle.call_args.args[0]()
        app.split.sashpos.assert_called_once_with(0, 240)
        app.initialize_split()
        self.assertEqual(app.root.after_idle.call_count, 1)

    def test_layout_results_and_persistence(self):
        import tkinter as tk
        import thesis_gui as gui
        with tempfile.TemporaryDirectory() as folder, patch.object(gui, 'ROOT', Path(folder)):
            root = tk.Tk()
            root.withdraw()
            app = gui.Application(root)
            root.update()
            self.assertEqual(len(app.tabs.tabs()), 4)
            self.assertNotIn('Ready to initialize', app.log.get('1.0', 'end'))
            app.set_status('failed', 'failure')
            self.assertEqual(str(app.status_label.cget('foreground')), '#FF6B78')
            app.set_status('cancelled', 'cancelled')
            self.assertEqual(str(app.status_label.cget('foreground')), '#F1C578')
            app.variables['Classification.Model'].set('RoBERTa')
            self.assertEqual(json.loads(app.settings_path.read_text())['Classification.Model'], 'RoBERTa')
            result = {'tables': {'Song label': [{'v26_label': 'Safe', 'v26_evidence_probability': .9,
                        'v26_evidence_window': 1, 'v26_window_count': 2}]},
                      'charts': [{'title': 'Erasure', 'labels': ['5%', '10%'], 'series': [{'name': 'Ranked', 'y': [.1, .2]}]}]}
            app.display(result)
            root.update()
            self.assertEqual(app.results.tab(0, 'text'), 'Prediction')
            app.history_dir.mkdir(parents=True)
            app.active_record = app.history_dir / 'test_run'
            app.active_record.mkdir()
            (app.active_record / 'request.json').write_text('{"action":"label"}')
            app.cancelled = False
            app.archive_run(json.dumps(result), 1.2)
            self.assertEqual(len(app.history_paths), 1)
            self.assertTrue((app.active_record / 'worker.log').exists())
            app.history_list.selection_set(0)
            app.load_history()
            self.assertIn('Completed', app.status.get())
            # Failed/cancelled runs must remain inspectable in history too.
            for cancelled, expected in ((False, 'Failed'), (True, 'Cancelled')):
                app.active_record = app.history_dir / expected
                app.active_record.mkdir()
                (app.active_record / 'request.json').write_text('{"action":"label"}')
                app.cancelled = cancelled
                app.archive_run(None, .5)
                self.assertEqual(json.loads((app.active_record / 'summary.json').read_text())['status'], expected)
            self.assertEqual(len(app.history_paths), 3)
            app.save_settings()
            # Reconstruct the application to verify a real settings reload.
            for child in root.winfo_children():
                child.destroy()
            restored = gui.Application(root)
            self.assertEqual(restored.variables['Classification.Model'].get(), 'RoBERTa')
            root.destroy()


if __name__ == '__main__':
    unittest.main()
