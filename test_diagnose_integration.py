"""CPU-only diagnostics integration; no checkpoint or reference model is loaded."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import diagnose as d


ROOT = Path(__file__).resolve().parent


class PanelInventoryTests(unittest.TestCase):
    def test_preparation_binds_exact_arm_ids_exposures_and_baseline_membership(self):
        common = []
        histories = {'control': [], 'treatment': []}
        source_rows = []
        for index in range(1572):
            name = f'common-{index:04d}'
            row = {'case_id': name, 'question': name, 'calls': [{'name': 'fresh', 'arguments': {}}],
                   'input_ids': [index + 1, 10000], 'prompt_tokens': 1,
                   'positions': {'control-1729': [index], 'treatment-1729': [index]}}
            common.append(row); source_rows.append(row)
        for arm in histories:
            for index in range(480):
                raw_id = f'pair-{index:04d}' + ('-control' if arm == 'control' else '')
                raw = {'case_id': raw_id, 'question': raw_id, 'calls': [{'name': 'history', 'arguments': {}}],
                       'model_input': f'prompt-{2000 + index}', 'before': {'sentinel': arm},
                       'group': f'group-{index % 8}'}
                histories[arm].append(raw)
                count = 4 if index < 132 else 3
                source_rows.append({'case_id': f'{arm}:{raw_id}', 'question': raw_id, 'calls': raw['calls'],
                                    'input_ids': [2000 + index, 10000], 'prompt_tokens': 1,
                                    'positions': {f'{arm}-1729': list(range(count))}})

        class Tokenizer:
            @classmethod
            def from_file(cls, path): return cls()
            def no_padding(self): pass
            def no_truncation(self): pass
            def decode(self, ids, **kwargs): return 'target' if ids == [10000] else f'prompt-{ids[0]}'
            def encode(self, text, **kwargs): return SimpleNamespace(ids=[int(text.split('-')[1])])

        prepared_requests = []

        class RPC:
            def __init__(self, script, registry): pass
            def __enter__(self): return self
            def __exit__(self, *error): pass
            def request(self, request, expected):
                prepared_requests.append(request)
                row = request['row']
                if request['source'] == 'history':
                    arm = row['case_id'].split(':')[0]
                    assert row['before']['sentinel'] == arm
                return {'kind': 'prepared', 'case': {'case_id': row['case_id'], 'source': request['source'],
                        'semantic_eligible': True, 'exact_target': 'target',
                        'prompt': request['prompt'], 'prompt_sha256': d.sha256(request['prompt'].encode()).hexdigest(),
                        'group': row.get('group', 'common')}}

        with TemporaryDirectory(dir=ROOT / '.runtime/tmp') as temporary:
            output = Path(temporary)
            source = {'directory': output / 'source', 'prepared': {'collection_receipt': {
                'control': {'path': 'control.jsonl'}, 'treatment': {'path': 'treatment.jsonl'}}},
                'manifest': {'assets': {'tokenizer': {'path': 'assets/tokenizer.json'}}},
                'train': output / 'frozen.jsonl', 'students': {
                    'baseline-7205': {}, 'control-1729': {}, 'treatment-1729': {}}}

            def selected_rows(path):
                path = Path(path)
                if path == source['train']: return iter(source_rows)
                if path.name == 'diagnostic-train-contexts.jsonl': return iter(())
                if path.name in ('control.jsonl', 'treatment.jsonl'):
                    return iter(histories[path.stem])
                raise AssertionError(f'Unexpected read: {path}')

            source['train'].write_text(''.join(d.canonical(row) + '\n' for row in source_rows))
            source['manifest']['assets']['tokenizer'] = d.record(ROOT / 'assets/tokenizer.json')
            with patch.object(d, 'rows', selected_rows), patch.object(d, 'NodeRPC', RPC), \
                    patch.object(d, 'validate_contexts', return_value=({}, {})), \
                    patch.object(d, 'panel_binding', return_value='inventory-fixture-only'), \
                    patch.dict(sys.modules, {'tokenizers': SimpleNamespace(Tokenizer=Tokenizer)}):
                result = d.prepare_panel(source, output, ROOT / 'assets/registry.json')
            saved = [json.loads(line) for line in (output / 'train-panel.jsonl').read_text().splitlines()]
            self.assertEqual(result['cases'], 2532)
            self.assertEqual(result['strata'], {'common': 1572, 'history': 960})
            self.assertEqual(len(set(result['qualification_ids'])), 32)
            self.assertEqual(result['qualification_ids'][0], max(row['case_id'] for row in source_rows))
            self.assertEqual(len(prepared_requests), 2532)
            by_id = {row['case']['case_id']: row for row in saved}
            control = by_id['control:pair-0000-control']
            self.assertEqual(control['exposure'], {'control-1729': 4, 'treatment-1729': 0, 'baseline-7205': 0})
            treatment = by_id['treatment:pair-0000']
            self.assertEqual(treatment['exposure'], {'control-1729': 0, 'treatment-1729': 4, 'baseline-7205': 0})
            self.assertEqual(by_id['common-0000']['exposure'],
                             {'control-1729': 1, 'treatment-1729': 1, 'baseline-7205': 1})
            self.assertTrue(all(row['baseline_exposure_count_is_membership_only'] for row in saved))
            self.assertTrue(all(row['input_token_count'] == 1 for row in saved))
            self.assertTrue(all('prompt' not in row['case'] for row in saved))
            self.assertTrue(all(row['storage'] == 'frozen-train-prefix-v1' for row in saved))
            with patch.dict(sys.modules, {'tokenizers': SimpleNamespace(Tokenizer=Tokenizer)}):
                expanded = d.panel_cases(result)
                self.assertEqual(len(expanded), 2532)
                self.assertTrue(all(row['case']['prompt'].startswith('prompt-') for row in expanded))
                damaged = dict(saved[0])
                damaged['case']['prompt_sha256'] = '0' * 64
                (output / 'train-panel.jsonl').write_text(d.canonical(damaged) + '\n')
                with self.assertRaisesRegex(ValueError, 'Restored TRAIN prompt'):
                    d.panel_cases({**result, 'file': d.record(output / 'train-panel.jsonl')})


class RealBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(str(ROOT / 'assets/tokenizer.json'))
        cls.tokenizer.no_padding(); cls.tokenizer.no_truncation()
        cls.context = next(d.rows(ROOT / 'assets/diagnostic-train-contexts.jsonl'))
        selected = {}
        for row in d.rows(ROOT / 'assets/train-full-v2.jsonl'):
            if row['case_id'] == cls.context['case_id'] or not selected:
                selected[row['case_id']] = row
            if len(selected) == 2 and cls.context['case_id'] in selected: break
        cls.source_rows = list(selected.values())

    def prepare(self, rpc, source_row):
        tokens = source_row['input_ids']
        boundary = source_row['prompt_tokens']
        prompt = self.tokenizer.decode(tokens[:boundary], skip_special_tokens=False)
        target = self.tokenizer.decode(tokens[boundary:], skip_special_tokens=False)
        self.assertEqual(self.tokenizer.encode(prompt, add_special_tokens=False).ids, tokens[:boundary])
        request = {'action': 'prepare', 'source': 'common', 'row': source_row, 'prompt': prompt}
        if source_row['case_id'] == self.context['case_id']:
            request['before'] = self.context['before']
        item = rpc.request(request, 'prepared')['case']
        self.assertEqual(item['exact_target'], target)
        self.assertTrue(item['semantic_eligible'])
        return item

    def test_real_tokenizer_node_bridge_scores_frozen_fresh_and_historical_targets(self):
        with d.NodeRPC('diagnose-data.mjs', ROOT / 'assets/registry.json') as rpc:
            for source_row in self.source_rows:
                item = self.prepare(rpc, source_row)
                generation = {'provider_kind': 'functiongemma', 'complete': True,
                              'raw_text': item['exact_target'], 'raw_ids': source_row['input_ids'][source_row['prompt_tokens']:]}
                observation = rpc.request({'action': 'score', 'case': item, 'generation': generation}, 'score')['observation']
                self.assertTrue(observation['success'])
                self.assertTrue(observation['native_exact_match'])
                self.assertTrue(observation['canonical_call_match'])
                self.assertEqual(observation['critical_status'], 'none_observed')

    def test_train_worker_resume_regrades_saved_raw_generation_without_regenerating(self):
        with TemporaryDirectory(dir=ROOT / '.runtime/tmp') as temporary, \
                d.NodeRPC('diagnose-data.mjs', ROOT / 'assets/registry.json') as rpc:
            item = self.prepare(rpc, self.source_rows[0])
            cases = [{'case': item, 'exposure': {'baseline-7205': 1}}]
            target = Path(temporary) / 'observations.jsonl'
            calls = []
            binding = 'a' * 64

            def generate(prompt):
                calls.append(prompt)
                return {'provider_kind': 'reference', 'complete': True, 'raw_text': item['exact_target'],
                        'raw_ids': [999999], 'finish_reason': 'complete_batch',
                        'diagnostic_worker_sha256': binding}

            d.train_worker(rpc, cases, target, generate, 'fixture', binding)
            self.assertEqual(len(calls), 1)
            first = target.read_bytes()
            with patch.object(d, 'append_observation', side_effect=AssertionError('Resume must not rewrite completed observations')):
                d.train_worker(rpc, cases, target, lambda prompt: self.fail('Resume regenerated an output'), 'fixture', binding)
            self.assertEqual(target.read_bytes(), first)
            with self.assertRaisesRegex(ValueError, 'different checkpoint or diagnostic worker'):
                d.train_worker(rpc, cases, target, lambda prompt: self.fail('Copied shard triggered generation'),
                               'different-checkpoint', 'b' * 64)
            self.assertEqual(target.read_bytes(), first)
            saved = json.loads(first)
            self.assertEqual(saved['generation']['raw_ids'], [999999])
            self.assertEqual(saved['generation']['diagnostic_worker_sha256'], binding)
            saved['generation']['raw_text'] = ''
            target.write_text(d.canonical(saved) + '\n')
            with self.assertRaisesRegex(ValueError, 'no longer reproduces'):
                d.train_worker(rpc, cases, target, generate, 'fixture', binding)

    def test_incomplete_student_batch_is_failure_with_unchanged_state(self):
        with d.NodeRPC('diagnose-data.mjs', ROOT / 'assets/registry.json') as rpc:
            item = self.prepare(rpc, self.source_rows[0])
            generation = {'provider_kind': 'functiongemma', 'complete': False,
                          'raw_text': item['exact_target'], 'raw_ids': [100] * 512, 'finish_reason': 'length'}
            result = rpc.request({'action': 'score', 'case': item, 'generation': generation}, 'score')['observation']
            self.assertFalse(result['success'])
            self.assertFalse(result['transport_valid'])
            self.assertFalse(result['native_exact_match'])
            self.assertEqual(result['after']['semantic'], item['before']['semantic'])
            self.assertEqual(result['after']['exchanges'][-1]['generation']['token_ids'], generation['raw_ids'])


if __name__ == '__main__':
    unittest.main()
