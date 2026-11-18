"""Regression tests for inspecting an inherited and completed DEV comparison."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import student_evidence as evidence


ROOT = Path(__file__).resolve().parent
SOURCE = Path('runs/actual-history-v3-4gpu/revisions/rope-alignment-v1/recovery/checkpoint-io-v1')
CHILD = SOURCE / 'completion-v1'
MODELS = ('baseline', 'control-1729', 'treatment-1729')


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


class CompletedPair:
    def __init__(self, root):
        self.root = root
        self.original_records = []
        self.completed_records = []
        self.checkpoints = {}
        self.score = {'observed_records': 1250, 'quality_valid': True, 'infrastructure_failures': 0,
                      'infrastructure_failure_case_ids': []}
        self.identities = {name: {'experiment_id': name} for name in MODELS}
        self.original_records.append(self.write(SOURCE / 'identities.json', self.identities))
        for name in MODELS:
            self.checkpoints[name] = self.write(Path('runs/weights') / name / 'checkpoint.safetensors', {'model': name})
        code = [self.write(path.relative_to(ROOT), {'name': path.stem}) for path in
                [ROOT / 'evaluate-actual.mjs', *sorted((ROOT / 'evaluation').glob('*.mjs'))]]
        assets = {key: self.write(Path('assets') / fname, {'asset': key}) for key, fname in
                  [('registry', 'registry.json'), ('history_dev', 'actual-history-v3-dev.json')]}
        assets['initial_checkpoint'] = self.checkpoints['baseline']
        manifest = {'assets': assets, 'code': {'files': code}, 'origin': {'checkpoint': self.checkpoints['baseline']}}
        manifest_entry = self.write(SOURCE / 'experiment.json', manifest)
        self.original_records.append(manifest_entry)
        self.parent_binding = digest(manifest)
        values = {}
        for name in MODELS:
            directory = SOURCE / 'evaluations' / name
            self.write(directory / 'input.json', {'known': False, 'run': self.identities[name],
                                                 'checkpoint': self.checkpoints[name]})
            value = {'score': self.score, 'observations': self.write(directory / 'observations.jsonl', {'arm': name}),
                     'checkpoint': self.checkpoints[name]}
            values[name] = value
        self.stage(SOURCE, 'baseline', values['baseline'])
        self.stage(SOURCE, 'control-1729-trained', {'checkpoint': self.checkpoints['control-1729']})
        child = {'source_directory': SOURCE.as_posix(), 'source_manifest': manifest_entry,
                 'source_snapshot': {'immutable': self.original_records}, 'policy': {'seed': 1729}}
        self.child_binding = digest(child)
        self.completed_records.append(self.write(CHILD / 'experiment.json', child))
        self.completed_records.append(self.write(CHILD / 'manifest-binding.json', {'sha256': self.child_binding}))
        self.stage(CHILD, 'control-1729-final', values['control-1729'])
        self.stage(CHILD, 'treatment-1729-trained', {'checkpoint': self.checkpoints['treatment-1729']})
        self.stage(CHILD, 'treatment-1729-final', values['treatment-1729'])
        self.report = {'identity_sha256': self.child_binding, 'baseline': values['baseline'],
                       'pairs': [{'seed': 1729, 'control': values['control-1729'], 'treatment': values['treatment-1729']}],
                       'TEST_opened': False, 'product_acceptance': False, 'status': 'inconclusive',
                       'completion_policy': child['policy'], 'exploratory': True, 'artifacts': self.completed_records}
        self.write(CHILD / 'result.json', self.report)

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, ensure_ascii=False) + '\n').encode()
        path.write_bytes(data)
        return {'path': path.relative_to(self.root).as_posix(), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}

    def stage(self, directory, name, value):
        source = directory == SOURCE
        binding = self.parent_binding if source else self.child_binding
        records = self.original_records if source else self.completed_records
        records.append(self.write(directory / 'stages' / (name + '.json'),
                                  {'identity_sha256': binding, 'result': value}))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        private = ROOT / '.runtime/tmp'
        private.mkdir(parents=True, exist_ok=True)
        temp = TemporaryDirectory(dir=private)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pair = CompletedPair(self.root)

    def test_completed_mixed_stage_locations_are_discovered_readonly(self):
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        registry, dev, arms = evidence.load_saved(self.root)
        self.assertEqual(registry, self.root / 'assets/registry.json')
        self.assertEqual(dev, self.root / 'assets/actual-history-v3-dev.json')
        self.assertEqual(set(arms), {'baseline-7205', 'control-1729', 'treatment-1729'})
        for name, (score, path) in arms.items():
            self.assertEqual(score, self.pair.score)
            arm = 'baseline' if name == 'baseline-7205' else name
            self.assertEqual(path, self.root / SOURCE / 'evaluations' / arm / 'observations.jsonl')
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_tampered_observation_bytes_are_rejected_even_if_parseable(self):
        self.pair.write(SOURCE / 'evaluations/treatment-1729/observations.jsonl', {'arm': 'control-1729'})
        with self.assertRaises(ValueError):
            evidence.load_saved(self.root)

    def test_missing_terminal_never_falls_back_to_old_budget_stopped_result(self):
        (self.root / CHILD / 'result.json').unlink()
        self.pair.write(SOURCE / 'result.json', {'status': 'budget_stopped'})
        with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
            evidence.load_saved(self.root)
        self.assertFalse((self.root / CHILD / 'result.json').exists())

    def test_wrong_arm_final_score_cannot_be_substituted_in_terminal(self):
        result = deepcopy(self.pair.report)
        result['pairs'][0]['control'] = result['pairs'][0]['treatment']
        self.pair.write(CHILD / 'result.json', result)
        with self.assertRaises(ValueError):
            evidence.load_saved(self.root)

    def test_replay_checks_nested_metrics_not_only_top_level_count(self):
        saved = dict(self.pair.score, panels={'challenge': {'critical_detected': 2}})
        evidence.validate_replay(saved, deepcopy(saved))
        other = deepcopy(saved)
        other['panels']['challenge']['critical_detected'] = 1
        with self.assertRaises(ValueError):
            evidence.validate_replay(saved, other)


if __name__ == '__main__':
    unittest.main()
