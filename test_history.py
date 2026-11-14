"""CPU-only interruption and decision tests; no model is loaded."""

from copy import deepcopy
import importlib.util
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import struct
from types import ModuleType, SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import history as h
import history_parity as parity
import history_prepare as prepare


class Clock:
    def __init__(self):
        self.value = 1000.0
    def __call__(self):
        return self.value


def metric(passed, planned):
    return {'passed': passed, 'planned': planned}


def fixture(passed, *, unresolved=0, reference=20, whole=0, critical=None):
    summary = {'supported': metric(passed, 100), 'outside_scope': metric(10, 10),
               'whole_dialogues': metric(whole, 10), 'standalone': metric(20, 20),
               'exact_reference': metric(1, 1), 'critical_review_required': unresolved,
               'critical_types': critical or {}, 'critical_detected': sum((critical or {}).values()), 'missing': 0}
    fixed = dict(deepcopy(summary), exact_reference=metric(reference, 100), critical_review_required=0,
                 critical_types={})
    return {'score': {'summary': summary, 'panels': {'closed_loop': summary, 'fixed_prefix': fixed}}, 'seconds': 1.0}


def observations(passed):
    return {f'case-{i}': {'case_id': f'case-{i}', 'family_id': f'family-{i % 10}',
        'panel': 'closed_loop', 'cohort': 'supported', 'success': i < passed,
        'critical_status': 'none_observed'} for i in range(100)}


class FakeBackend:
    def __init__(self, root, budget, *, fail_at=None, unresolved=0):
        self.root, self.budget, self.fail_at = root, budget, fail_at
        self.calls, self.unresolved = [], unresolved
    def after(self, name, value):
        self.calls.append(name)
        if name == self.fail_at:
            self.fail_at = None
            raise OSError('simulated interruption after atomic output')
        return value
    def checkpoint(self, name, step):
        path = self.root / f'{name}-{step}.bin'
        h.immutable(path, {'name': name, 'step': step})
        return h.file_record(path, relative_to=self.root)
    def validate_artifact(self, record):
        if h.file_record(self.root / record['path'], relative_to=self.root) != record:
            raise ValueError('changed checkpoint')
    def pilot(self, deadline):
        checkpoint = self.root / 'pilot-7207.bin'
        if not checkpoint.exists():
            self.budget.claim_updates(6)
        value = {'checkpoint': self.checkpoint('pilot', 7207),
                 'measurements': [{'seconds': 1.0}, {'seconds': 1.0}]}
        return self.after('pilot', value)
    def remaining_updates(self, name):
        if (self.root / f'{name}-7991.bin').exists():
            return 0
        return 393 if (self.root / f'{name}-7598.bin').exists() else 786
    def train(self, name, milestone, deadline):
        filename = self.root / f'{name}-{7205 + milestone}.bin'
        if not filename.exists():
            done = 786 - self.remaining_updates(name)
            self.budget.claim_updates(milestone - done)
        value = {'checkpoint': self.checkpoint(name, 7205 + milestone), 'step': 7205 + milestone}
        return self.after(f'{name}-{milestone}', value)
    def evaluate(self, name, checkpoint, deadline, *, known=False):
        passed = 20 if name == 'baseline' else 30 if name.startswith('control') else 60
        value = fixture(passed, unresolved=self.unresolved if name.startswith('treatment') else 0,
                        reference=20 if name == 'baseline' else 30 if name.startswith('control') else 70,
                        whole=4 if name.startswith('treatment') else 0)
        value['checkpoint'] = h.file_record(checkpoint, relative_to=self.root)
        path = self.root / f'{name}-{known}-observations.json'
        h.immutable(path, observations(passed))
        value['observations'] = h.file_record(path, relative_to=self.root)
        return self.after(f'{name}-{"regression" if known else "final"}', value)
    def validate_evaluation(self, value, *, known=False):
        self.validate_artifact(value['checkpoint'])
        self.validate_artifact(value['observations'])
    def outcomes(self, value):
        return h.read(self.root / value['observations']['path'])
    def space(self, evaluations):
        pass


class BudgetTests(unittest.TestCase):
    def test_precharge_and_frozen_limit(self):
        with TemporaryDirectory() as directory:
            path, clock = Path(directory) / 'budget.json', Clock()
            budget = h.Budget(path, 'identity', clock=clock)
            budget.claim_updates(h.MAX_UPDATES)
            with self.assertRaises(RuntimeError):
                budget.claim_updates(1)
            with self.assertRaises(ValueError):
                h.Budget(path, 'new-identity', clock=clock)
            with self.assertRaises(ValueError):
                h.Budget(path, 'identity', hours=35, clock=clock)

    def test_interrupted_phase_is_charged_without_reset(self):
        with TemporaryDirectory() as directory:
            path, clock = Path(directory) / 'budget.json', Clock()
            budget = h.Budget(path, 'identity', clock=clock)
            record = budget.account['record']
            record['inflight'] = {'name': 'evaluation', 'started_at': clock()}
            h.replace_json(path, record)
            clock.value += 125
            resumed = h.Budget(path, 'identity', clock=clock)
            self.assertEqual(resumed.account['record']['elapsed_seconds'], 125)
            with resumed.phase('finish'):
                clock.value += 5
                resumed.claim_updates(2)
                with self.assertRaises(RuntimeError):
                    with resumed.phase('parallel'):
                        pass
            self.assertEqual(resumed.account['record']['elapsed_seconds'], 130)
            self.assertEqual(h.read(path)['charged_updates'], 2)


class GateTests(unittest.TestCase):
    def values(self):
        return fixture(20), fixture(30, reference=30), fixture(60, reference=70, whole=4)
    def test_gain_passes_and_unresolved_blocks(self):
        b, a, t = self.values()
        self.assertTrue(h.decision(b, a, t, (observations(30), observations(60)))['passed'])
        t['score']['panels']['fixed_prefix']['critical_review_required'] = 1
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertEqual(result['status'], 'inconclusive')
        self.assertFalse(result['passed'])
    def test_challenge_critical_blocks_and_kind_offset_cannot_hide(self):
        b, a, t = self.values()
        for value in (b, a, t):
            value['score']['panels']['challenge'] = deepcopy(value['score']['panels']['fixed_prefix'])
        a['score']['panels']['challenge']['critical_types'] = {'wrong_answer': 5}
        t['score']['panels']['challenge']['critical_types'] = {'fabrication': 1}
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertFalse(result['gates']['other_critical'])
    def test_denominator_or_family_mutation_fails(self):
        b, a, t = self.values()
        t['score']['summary']['supported']['planned'] = 99
        with self.assertRaises(ValueError):
            h.decision(b, a, t, (observations(30), observations(60)))
        changed = observations(60)
        changed['case-0']['family_id'] = 'different'
        with self.assertRaises(ValueError):
            h.paired_bootstrap(observations(30), changed, repeats=50)
    def test_bootstrap_samples_whole_families(self):
        result = h.paired_bootstrap(observations(30), observations(60), repeats=500)
        self.assertEqual(result['families'], 10)
        self.assertEqual(result['lower'], .3)
        self.assertEqual(result['upper'], .3)
    def test_bootstrap_joins_overlapping_families_without_splitting(self):
        left, right = observations(30), observations(60)
        for records in (left, right):
            for row in records.values():
                row['bootstrap_family_id'] = f"block-{int(row['family_id'].split('-')[1]) // 2}"
        self.assertEqual(h.paired_bootstrap(left, right, repeats=50)['families'], 5)
        for records in (left, right):
            records['case-0']['bootstrap_family_id'] = 'split-original-family'
        with self.assertRaises(ValueError):
            h.paired_bootstrap(left, right, repeats=50)
    def test_infrastructure_failure_blocks_gain_but_overflow_stays_in_denominator(self):
        rows = [{'case_id': 'overflow', 'generation': {'error': 'context_overflow'}},
                {'case_id': 'broken', 'after': {'exchanges': [{'outcome': 'provider_error', 'error_code': 'inference_failed'}]}},
                {'case_id': 'model-error', 'after': {'exchanges': [{'outcome': 'error', 'error_code': 'invalid_argument'}]}}]
        self.assertEqual(h.infrastructure_failures(rows), ['broken'])
        b, a, t = self.values()
        a['score']['infrastructure_failures'] = 1
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertFalse(result['gates']['infrastructure_clean'])
        self.assertEqual(result['status'], 'inconclusive')
    def test_terminal_summary_does_not_dump_raw_results(self):
        baseline = fixture(20)
        baseline['score']['raw_results'] = 'large-raw-content' * 10000
        result = {'status': 'inconclusive', 'baseline': baseline, 'pairs': [], 'selection': None}
        stream = io.StringIO()
        with redirect_stdout(stream):
            h.print_summary(result, Path('/tmp/history-test-result.json'))
        self.assertLess(len(stream.getvalue()), 1000)
        self.assertNotIn('large-raw-content', stream.getvalue())
        self.assertIn('/tmp/history-test-result.json', stream.getvalue())


class BridgeTests(unittest.TestCase):
    def test_resumed_timing_profile_excludes_offline_gap_and_keeps_completed_work(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'timings.jsonl'
            h.append_timing(path, 'load', 2.0)
            h.append_timing(path, 'generation', 5.0, case_id='first')
            h.append_timing(path, 'overhead', 1.0)
            preserved = path.read_bytes()
            with path.open('ab') as stream:
                stream.write(b'{"interrupted":')
            self.assertEqual(sum(row['seconds'] for row in h.timing_records(path, repair=True)), 8.0)
            self.assertEqual(path.read_bytes(), preserved)
            with patch.object(h, 'time', return_value=100000000.0):
                h.append_timing(path, 'load', 3.0)
                h.append_timing(path, 'generation', 6.0, case_id='last')
            self.assertEqual(sum(row['seconds'] for row in h.timing_records(path)), 17.0)

    def test_native_generate_api_and_overflow_preserve_attempts(self):
        class Input(io.StringIO):
            def close(self):
                self.saved = self.getvalue()
                super().close()
        class Process:
            def __init__(self):
                self.stdin = Input()
                self.stdout = io.StringIO('\n'.join(json.dumps(value) for value in [
                    {'kind': 'generate', 'case_id': 'fits', 'prompt': 'short'},
                    {'kind': 'generate', 'case_id': 'overflow', 'prompt': 'long'},
                    {'kind': 'result', 'score': {'observed_records': 2}},
                ]))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def wait(self):
                return 0
            def kill(self):
                pass
        class Tokenizer:
            @classmethod
            def from_file(cls, path):
                return cls()
            def no_padding(self):
                pass
            def no_truncation(self):
                pass
            def encode(self, text, *, add_special_tokens):
                return SimpleNamespace(ids=[2] * (32768 if text == 'long' else 3))
            def decode(self, ids, *, skip_special_tokens):
                return '<start_function_response>'
        class Runtime:
            calls = 0
            def generate(self, prompt, *, max_new_tokens, stop_ids):
                assert len(prompt) == 3 and max_new_tokens == 512 and stop_ids == {50}
                self.calls += 1
                return [50]
        process, runtime = Process(), Runtime()
        with TemporaryDirectory() as temporary, patch.dict(sys.modules, {'tokenizers': SimpleNamespace(Tokenizer=Tokenizer)}), \
                patch.object(h.subprocess, 'Popen', return_value=process):
            score = h.evaluate_bridge(runtime, {'tokenizer': ('tokenizer', {}), 'registry': ('registry', {})},
                                      'panel', Path(temporary) / 'observations.jsonl', {}, h.time() + 60)
        self.assertEqual(runtime.calls, 1)
        replies = [json.loads(line) for line in process.stdin.saved.splitlines()]
        self.assertEqual(replies[1]['error'], 'context_overflow')
        self.assertEqual(score['observed_records'], 2)


class SourceTests(unittest.TestCase):
    def source(self, root, *, missing_optimizer=False):
        import ledger
        runs = root / 'runs'
        runs.mkdir()
        assets = {}
        for name in ('base', 'tokenizer', 'initial_checkpoint'):
            path = root / name
            path.write_text(name)
            assets[name] = (path, h.file_record(path, relative_to=root))
        identity = {'experiment_id': 'full-course', 'architecture_id': 'functiongemma270m-lora16-volta-fp32-v2',
                    'recipe': {'initial_step': 4847, 'examples_per_update': 4, 'training': {'updates': 2358}},
                    'code': {'files': [{'path': 'run.py', 'sha256': '0' * 64}]}, 'runtime': {'test': True},
                    'train': {'sha256': '0' * 64}, 'evaluation': {'sha256': '0' * 64, 'split': 'DEV-VOLUME'},
                    **{key: value[1] for key, value in assets.items()}}
        lineage = {'identity_sha256': h.digest(identity), 'initial_checkpoint_sha256': assets['initial_checkpoint'][1]['sha256']}
        course_id = 'full-train-v2-4gpu-full-course'
        with ledger.open_run(runs, course_id, identity) as run:
            checkpoint = run.path / 'checkpoint-step-7205.safetensors'
            header = {'__metadata__': {'step': '7205', 'lineage': json.dumps(lineage)}}
            for index in range(252):
                for prefix in ('lora', 'adafactor.row', 'adafactor.col'):
                    if missing_optimizer and index == 0 and prefix == 'adafactor.col':
                        continue
                    header[f'{prefix}.tensor{index}'] = {'dtype': 'F32'}
            raw = json.dumps(header).encode()
            checkpoint.write_bytes(struct.pack('<Q', len(raw)) + raw)
            observations_path = run.path / 'original-observations.jsonl'
            observations_path.write_text('{}\n' * 155)
            evaluated = run.write_result({'missing': 0}, h.file_record(checkpoint), name='result-step-7205.json',
                       step=7205, observed_records=155, observations=h.file_record(observations_path))
        outer = runs / 'full-train-v2-4gpu'
        outer.mkdir()
        h.immutable(outer / 'result.json', {'status': 'complete', 'completed_epochs': 3,
                   'epochs': [{'details': {'step': 5633}}, {'details': {'step': 6419}}, evaluated]})
        return runs, assets, checkpoint
    def test_real_source_receipt_and_optimizer_are_mandatory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, assets, checkpoint = self.source(root)
            with patch.object(h, 'ROOT', root):
                path, receipt = h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)
                self.assertEqual(path, checkpoint)
                self.assertEqual(receipt['step'], 7205)
                checkpoint.rename(checkpoint.with_name('checkpoint-step-6419.safetensors'))
                with self.assertRaises(FileNotFoundError):
                    h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, assets, _ = self.source(root, missing_optimizer=True)
            with patch.object(h, 'ROOT', root), self.assertRaisesRegex(ValueError, 'optimizer'):
                h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)


class PanelContractTests(unittest.TestCase):
    def test_real_frozen_panel_schema_and_all_planned_denominators(self):
        panel_root = REPO.parents[1] / 'data/corpus/revisions/actual-history-v3/dev'
        bundle_path = panel_root / 'actual-history-v3-dev.json'
        bundle = h.read(bundle_path)
        gold = h.read(panel_root / 'gold-replay.json')
        config = h.read(REPO / 'experiments-history.json')
        config['assets'] = {'history_dev': {'path': bundle_path.name}}
        rows = []
        for panel in ('fixed_prefix', 'closed_loop', 'challenge'):
            for authored in bundle['panels'][panel]:
                row = {key: authored.get(key) for key in ('case_id', 'panel', 'cohort', 'family_id',
                       'bootstrap_family_id', 'reference_eligible', 'dialogue_id')}
                row.update(success=True, critical_status='none_observed', critical_category=None,
                           reference_success=True if row['reference_eligible'] else None)
                rows.append(row)
        score = {'summary': gold['panels']['closed_loop'], 'panels': gold['panels'], 'observed_records': len(rows)}
        with TemporaryDirectory() as temporary, patch.object(h, 'ROOT', panel_root):
            path = Path(temporary) / 'observations.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            h.validate_score(score, path, config)
            self.assertEqual(len(rows), 1250)
            rows[0]['family_id'] = 'changed-family'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                h.validate_score(score, path, config)


class ResumeTests(unittest.TestCase):
    def execute(self, directory, fail_at=None, unresolved=0):
        budget = h.Budget(directory / 'budget.json', 'binding')
        stages = h.Stages(directory, 'binding', budget)
        backend = FakeBackend(directory, budget, fail_at=fail_at, unresolved=unresolved)
        initial = directory / 'initial.bin'
        h.immutable(initial, {'step': 7205})
        with patch.object(h, 'ROOT', directory):
            result = h.orchestrate(backend, stages, budget, initial)
        return result, budget, backend
    def test_resume_after_every_model_phase(self):
        phases = ['pilot', 'baseline-final']
        for seed in h.SEEDS:
            for arm in ('control', 'treatment'):
                name = f'{arm}-{seed}'
                phases.extend([f'{name}-393', f'{name}-regression', f'{name}-786', f'{name}-final'])
        for phase in phases:
            with self.subTest(phase=phase), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with self.assertRaises(OSError):
                    self.execute(directory, fail_at=phase)
                preserved = {p: p.read_bytes() for p in directory.glob('*.bin')}
                result, budget, _ = self.execute(directory)
                self.assertEqual(result['status'], 'confirmed_diagnostic_gain')
                self.assertEqual(budget.account['record']['charged_updates'], 3150)
                self.assertTrue(all(path.read_bytes() == raw for path, raw in preserved.items()))
                repeated, _, backend = self.execute(directory)
                self.assertEqual(repeated, result)
                self.assertEqual(backend.calls, [])
    def test_unresolved_stops_second_seed(self):
        with TemporaryDirectory() as temporary:
            result, budget, backend = self.execute(Path(temporary), unresolved=1)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertEqual(len(result['pairs']), 1)
            self.assertEqual(budget.account['record']['charged_updates'], 1578)
            self.assertFalse(any('2718' in call for call in backend.calls))
    def test_changed_saved_checkpoint_is_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.execute(directory)
            (directory / 'control-1729-7598.bin').write_text('changed')
            with self.assertRaises(ValueError):
                self.execute(directory)
    def test_source_paths_cannot_leave_repository(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                h.contained(root.parent / 'other-user', root)
            (root / 'alias').symlink_to(root.parent)
            with self.assertRaises(ValueError):
                h.contained(root / 'alias' / 'outside', root)


class ParityRecoveryTests(unittest.TestCase):
    def test_io_retry_cannot_hide_later_numerical_failure_and_constructors_are_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = {}
            for name in ('base', 'tokenizer', 'history_replay', 'history_template'):
                assets[name] = root / name
                assets[name].write_text(name)
            checkpoint = root / 'checkpoint'
            checkpoint.write_text('checkpoint')
            model_path = root / 'model.py'
            model_path.write_text('fake model module')
            fake_torch = ModuleType('torch')
            fake_torch.cuda = SimpleNamespace(empty_cache=lambda: None)
            fake_nn = ModuleType('torch.nn')
            fake_nn.functional = SimpleNamespace()
            fake_model = ModuleType('model')
            fake_model.__file__ = str(model_path)
            attempts, closed = [], []
            def construct(*args, **kwargs):
                attempts.append('runtime')
                if len(attempts) == 1:
                    raise OSError('fixture transient I/O failure')
                return SimpleNamespace(close=lambda: closed.append(True))
            fake_model.Runtime = construct
            tokenizer = SimpleNamespace(no_truncation=lambda: None, no_padding=lambda: None,
                         encode=lambda text, **kwargs: SimpleNamespace(ids=[2 if text == '<bos>' else 50]))
            fake_tokens = ModuleType('tokenizers')
            fake_tokens.Tokenizer = SimpleNamespace(from_file=lambda path: tokenizer)
            output = root / 'parity'
            imports = {'torch': fake_torch, 'torch.nn': fake_nn, 'model': fake_model, 'tokenizers': fake_tokens}
            with patch.dict(sys.modules, imports), patch.object(parity, 'version', return_value='4.57.3'), \
                    patch.object(parity, 'selected_probes', return_value=[{}]), \
                    patch.object(parity, 'template_check', return_value=[]), \
                    patch.object(parity, 'official_model', side_effect=AssertionError('fixture numerical failure')):
                budget = SimpleNamespace(check=lambda: None)
                with self.assertRaises(OSError):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                with self.assertRaises(AssertionError):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                self.assertEqual(len(list(output.glob('parity-failed*.json'))), 2)
                self.assertEqual(len(closed), 1)
                with self.assertRaisesRegex(RuntimeError, 'previously failed'):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(len(closed), 1)


class CollectionRecoveryTests(unittest.TestCase):
    def test_remaining_estimate_counts_only_completed_bound_prefixes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipts.jsonl'
            specs = [{'prefix_id': 'a', 'prefix': [1, 2]}, {'prefix_id': 'b', 'prefix': [3]}]
            fingerprint = {'checkpoint_step': 7205, 'checkpoint_sha256': 'frozen'}
            first = {'prefix_id': 'a', 'model_fingerprint': fingerprint, 'observations': [1, 2]}
            path.write_text(json.dumps(first) + '\n' + '{"partial":')
            self.assertEqual(prepare.remaining_requests(specs, path, fingerprint), 1)
            second = {'prefix_id': 'b', 'model_fingerprint': fingerprint, 'observations': [3]}
            path.write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
            self.assertEqual(prepare.remaining_requests(specs, path, fingerprint), 0)
            with self.assertRaises(ValueError):
                prepare.remaining_requests(specs, path, {'checkpoint_step': 6419})
    def test_provenance_artifacts_and_actual_origin_are_required(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            saved = {}
            for key in ('control', 'treatment', 'prefix_receipts'):
                path = directory / f'{key}.jsonl'
                path.write_text(key)
                saved[key] = prepare.file_record(path, relative_to=directory)
            saved['collection'] = {'rows': 480, 'control_rows': 480, 'actual_observed_rows': 480,
                 'reconstructed_rows': 0, 'sha256': saved['treatment']['sha256'],
                 'control_sha256': saved['control']['sha256']}
            prepare.verify_collection(directory, saved)
            saved['collection']['actual_observed_rows'] = 0
            with self.assertRaises(ValueError):
                prepare.verify_collection(directory, saved)
            saved['collection']['actual_observed_rows'] = 480
            (directory / 'prefix_receipts.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                prepare.verify_collection(directory, saved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
