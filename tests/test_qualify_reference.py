"""Offline controls for the isolated 32-case Qwen qualification."""

from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import qualify_reference as q

REPO = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory(dir=q.ROOT / '.runtime/tmp')))
        for module in (q, q.diagnostic, q.inspector):
            self.stack.enter_context(patch.object(module, 'ROOT', self.root))
        self.stack.enter_context(patch.object(q.diagnostic, 'ACTIVE_BUDGET', None))
        self.stack.enter_context(patch.object(q.diagnostic, 'require_space'))
        self.parent = self.root / 'runs' / q.SOURCE_ID
        self.parent.mkdir(parents=True)
        self.child = self.root / q.CHILD
        self.child.mkdir(parents=True)

    def tearDown(self):
        self.stack.close()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')

    @staticmethod
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))

    @staticmethod
    def case(index):
        prompt = f'frozen prompt {index}'
        return {'case': {'case_id': f'case-{index:02d}', 'case_sha256': f'case-hash-{index}',
                         'prompt': prompt, 'prompt_sha256': sha256(prompt.encode()).hexdigest(),
                         'semantic_eligible': True, 'cohort': 'supported', 'passes': index == 0}}

    @staticmethod
    def raw(item, valid=True):
        text = '<tool_call>{"name":"explain_purpose","arguments":{"subject":"TR"}}</tool_call>'
        if not valid:
            text = 'Here is an explanation instead of a function call.'
        return {'raw_text': text + '<|im_end|>', 'raw_token_ids': [12, 151645],
                'finish_reason': 'eos', 'eos_token_id': 151645, 'eos_token_text': '<|im_end|>',
                'generated_tokens': 2, 'max_new_tokens': 2048, 'input_tokens': 25,
                'input_sha256': item['prompt_sha256'], 'rendered_prompt_sha256': 'a' * 64}

    def generation(self, item, binding='worker', valid=True):
        return q.converted_generation(self.raw(item, valid), item, binding)

    @staticmethod
    def receipt(item, generation):
        passed = generation['complete'] and item.get('passes', False)
        return {'case_id': item['case_id'], 'case_sha256': item['case_sha256'],
                'prompt_sha256': item['prompt_sha256'], 'generation': deepcopy(generation),
                'semantic_eligible': item.get('semantic_eligible', True), 'cohort': item['cohort'],
                'transport_valid': generation['complete'], 'success': passed,
                'critical_status': item.get('critical_status', 'none_observed'),
                'native_exact_match': passed, 'canonical_call_match': passed}

    def rpc(self):
        fixture = self

        class RPC:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def request(self, request, expected):
                assert expected == 'score' and request['action'] == 'score' and request['compact']
                return {'observation': fixture.receipt(request['case'], request['generation'])}

        return RPC()


class GenerationTests(Fixture):
    def test_conversion_preserves_raw_and_changes_only_serialization(self):
        item = self.case(0)['case']
        original = self.raw(item)
        generation = q.converted_generation(original, item, 'worker')
        q.verify_generation(generation, item, 'worker')
        self.assertEqual(generation['reference_raw_text'], original['raw_text'])
        self.assertEqual(generation['raw_ids'], original['raw_token_ids'])
        self.assertEqual(generation['raw_text'],
                         '<start_function_call>call:explain_purpose{subject:<escape>TR<escape>}'
                         '<end_function_call><start_function_response>')
        self.assertNotEqual(generation['raw_text'], original['raw_text'])

    def test_saved_raw_conversion_prompt_and_worker_tampering_are_rejected(self):
        item = self.case(0)['case']
        base = self.generation(item)
        changes = {'native_text': 'repaired output', 'raw_text': 'repaired output',
                   'reference_raw_text': 'different original<|im_end|>', 'raw_ids': [12, 99],
                   'input_sha256': 'wrong prompt', 'diagnostic_worker_sha256': 'wrong worker',
                   'transport_recipe_sha256': 'wrong recipe', 'generated_tokens': 99,
                   'max_new_tokens': 4096, 'rendered_prompt_sha256': 'short'}
        for key, value in changes.items():
            with self.subTest(field=key):
                generation = deepcopy(base)
                generation[key] = value
                with self.assertRaises(ValueError):
                    q.verify_generation(generation, item, 'worker')

    def test_invalid_and_incomplete_generations_remain_planned_failures(self):
        item = self.case(0)['case']
        for reason in ('eos', 'length'):
            with self.subTest(reason=reason):
                raw = self.raw(item, valid=False)
                raw['finish_reason'] = reason
                generation = q.converted_generation(raw, item, 'worker')
                q.verify_generation(generation, item, 'worker')
                self.assertFalse(generation['complete'])
                self.assertEqual(generation['raw_text'], raw['raw_text'])
                self.assertEqual(generation['native_text'], '')

    def test_overflow_must_be_real_and_have_no_partial_output(self):
        item = self.case(0)['case']
        generation = {'error': 'context_overflow', 'raw_text': '', 'raw_ids': [], 'complete': False,
                      'provider_kind': 'qwen3-hermes', 'input_tokens': 32760, 'max_new_tokens': 2048,
                      'context_tokens': q.reference.CONTEXT_TOKENS,
                      'input_sha256': item['prompt_sha256'], 'diagnostic_worker_sha256': 'worker',
                      'transport_recipe_sha256': q.inspector.digest(q.transport.RECIPE)}
        q.verify_generation(generation, item, 'worker')
        for field, changed in (('input_tokens', 2), ('raw_text', 'partial'), ('raw_ids', [3]),
                               ('context_tokens', 65536), ('complete', True)):
            altered = {**generation, field: changed}
            with self.subTest(field=field), self.assertRaises(ValueError):
                q.verify_generation(altered, item, 'worker')


class SelectionTests(Fixture):
    def panel(self):
        selected = [f'case-{index:02d}' for index in range(32)]
        source, prepared, prompts, token_ids = [], [], {}, {}
        for index in range(34):
            envelope = self.case(index)
            item = envelope['case']
            prompt = item.pop('prompt')
            target = f'target-{index}'
            prompts[index] = prompt
            prompts[1000 + index] = target
            token_ids[prompt] = [index]
            calls = [{'name': 'f', 'arguments': {'x': index}}]
            item.update(gold_calls=calls, exact_target=target)
            envelope['storage'] = 'frozen-train-prefix-v1'
            source.append({'case_id': item['case_id'], 'input_ids': [index, 1000 + index],
                           'prompt_tokens': 1, 'calls': calls})
            prepared.append(envelope)
        train, panel, tokenizer = (self.parent / name for name in ('train.jsonl', 'panel.jsonl', 'tokenizer.json'))
        self.write_rows(train, source)
        self.write_rows(panel, prepared)
        tokenizer.write_text('{}')
        fake = Mock()
        fake.decode.side_effect = lambda ids, **kwargs: ''.join(prompts[index] for index in ids)
        fake.encode.side_effect = lambda text, **kwargs: SimpleNamespace(ids=token_ids[text])
        self.stack.enter_context(patch.dict(sys.modules, {'tokenizers': SimpleNamespace(
            Tokenizer=SimpleNamespace(from_file=Mock(return_value=fake)))}))
        self.stack.enter_context(patch.object(q.transport, 'adapt_prompt', return_value=[]))
        result = {'qualification_ids': selected, 'source_train': q.diagnostic.record(train),
                  'file': q.diagnostic.record(panel), 'tokenizer': q.diagnostic.record(tokenizer)}
        return result, source, prepared, fake

    def test_only_original_32_are_restored_in_original_order(self):
        panel, _, _, tokenizer = self.panel()
        before = {path: path.read_bytes() for path in self.parent.iterdir() if path.is_file()}
        cases = q.select_cases(panel)
        self.assertEqual([row['case']['case_id'] for row in cases], panel['qualification_ids'])
        self.assertEqual(tokenizer.decode.call_count, 64)
        self.assertEqual(tokenizer.encode.call_count, 32)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_missing_duplicate_or_changed_prepared_cases_are_rejected(self):
        panel, _, prepared, _ = self.panel()
        path = self.root / panel['file']['path']
        mutations = [prepared[1:], prepared + [deepcopy(prepared[0])], deepcopy(prepared)]
        mutations[-1][0]['case']['prompt_sha256'] = 'altered'
        for index, rows in enumerate(mutations):
            self.write_rows(path, rows)
            current = {**panel, 'file': q.diagnostic.record(path)}
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                q.select_cases(current)

    def test_changed_source_hash_or_duplicate_selected_ids_are_rejected(self):
        panel, source, _, _ = self.panel()
        for ids in (panel['qualification_ids'][:-1], panel['qualification_ids'][:-1] + panel['qualification_ids'][:1]):
            with self.subTest(count=len(ids)), self.assertRaises(ValueError):
                q.select_cases({**panel, 'qualification_ids': ids})
        source[0]['calls'] = [{'name': 'wrong', 'arguments': {}}]
        path = self.root / panel['source_train']['path']
        self.write_rows(path, source)
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            q.select_cases(panel)
        with self.assertRaisesRegex(ValueError, 'prompt or target'):
            q.select_cases({**panel, 'source_train': q.diagnostic.record(path)})


class ResumeTests(Fixture):
    def test_partial_resume_generates_only_missing_cases_and_preserves_closed_files(self):
        cases = [self.case(index) for index in range(3)]
        path = self.child / 'gpu-0.jsonl'
        first = self.receipt(cases[0]['case'], self.generation(cases[0]['case']))
        self.write_rows(path, [first])
        closed = self.parent / 'result.json'
        closed.write_text('{"status":"closed"}\n')
        original = closed.read_bytes()
        generate = Mock(side_effect=lambda item: self.generation(item))
        with redirect_stdout(io.StringIO()):
            q.run_shard(cases, path, self.rpc(), generate, 'worker')
        self.assertEqual([call.args[0]['case_id'] for call in generate.call_args_list], ['case-01', 'case-02'])
        before = path.read_bytes()
        generate.reset_mock()
        q.run_shard(cases, path, self.rpc(), generate, 'worker')
        generate.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(closed.read_bytes(), original)

    def test_wrong_saved_prefix_stops_before_generating(self):
        cases = [self.case(index) for index in range(2)]
        path = self.child / 'gpu-0.jsonl'
        wrong = self.receipt(cases[1]['case'], self.generation(cases[1]['case']))
        self.write_rows(path, [wrong])
        generate = Mock()
        with self.assertRaisesRegex(ValueError, 'case order or identity'):
            q.run_shard(cases, path, self.rpc(), generate, 'worker')
        generate.assert_not_called()

    def test_only_uncommitted_new_shard_tail_is_recovered(self):
        cases = [self.case(index) for index in range(2)]
        path = self.child / 'gpu-0.jsonl'
        self.write_rows(path, [self.receipt(cases[0]['case'], self.generation(cases[0]['case']))])
        committed = path.read_bytes()
        with path.open('ab') as stream:
            stream.write(b'{"interrupted":')
        generate = Mock(side_effect=lambda item: self.generation(item))
        with redirect_stdout(io.StringIO()):
            q.run_shard(cases, path, self.rpc(), generate, 'worker')
        self.assertTrue(path.read_bytes().startswith(committed))
        self.assertEqual(generate.call_count, 1)

    def test_hardlinked_output_is_rejected_before_recovery_or_truncation(self):
        target = self.parent / 'closed-observations.jsonl'
        target.write_bytes(b'{"closed":true}\n{"partial":')
        before = target.read_bytes()
        output = self.child / 'gpu-0.jsonl'
        output.hardlink_to(target)
        generate = Mock()
        with patch.object(q.diagnostic, 'recover_jsonl', side_effect=AssertionError('Do not repair aliased file')):
            with self.assertRaisesRegex(ValueError, 'must not alias'):
                q.run_shard([self.case(0)], output, self.rpc(), generate, 'worker')
        generate.assert_not_called()
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(output.read_bytes(), before)


class PrivateOutputTests(Fixture):
    def test_symlink_ancestor_is_rejected_without_touching_target(self):
        self.child.rmdir()
        self.child.parent.rmdir()
        target = self.parent / 'untouched'
        target.mkdir()
        marker = target / 'keep.json'
        marker.write_text('{"kept":true}\n')
        self.child.parent.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real private directories'):
            q.output_directory()
        self.assertEqual(marker.read_text(), '{"kept":true}\n')
        self.assertEqual({path.name for path in target.iterdir()}, {'keep.json'})

    def test_symlink_child_file_or_directory_is_rejected_without_touching_target(self):
        target = self.parent / 'closed'
        target.mkdir()
        marker = target / 'keep.json'
        marker.write_text('{"kept":true}\n')
        for name, destination in (('gpu-0.jsonl', marker), ('nested', target)):
            link = self.child / name
            link.symlink_to(destination, target_is_directory=destination.is_dir())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'must not redirect'):
                q.output_directory()
            self.assertEqual(marker.read_text(), '{"kept":true}\n')
            link.unlink()
        self.assertEqual(q.output_directory(), self.child)

    def unbound_manifest(self):
        cases = [self.case(index) for index in range(32)]
        source = {'parent': self.parent, 'evidence': [], 'identity': {'code': {'files': []}},
                  'manifest': {'reference': {'frozen': True}, 'train_panel': {}}}
        code = {'files': [], 'git_commit': 'published', 'dirty': False}
        manifest = {'schema': 1, 'revision': q.REVISION, 'source_evidence': [],
                    'reference': source['manifest']['reference'], 'recipe': q.transport.RECIPE,
                    'case_ids': [row['case']['case_id'] for row in cases],
                    'case_hashes': [row['case']['case_sha256'] for row in cases],
                    'code': code, 'hardware': {'devices': [0, 1, 2, 3]}, 'dependencies': {'pinned': True},
                    'policy': {'cases': 32, 'gpus': 4, 'format_minimum': 30,
                               'supported_minimum': 1, 'training': False, 'dev': False}}
        self.write(self.child / 'experiment.json', manifest)
        self.stack.enter_context(patch.object(q, 'code_identity', return_value=code))
        return manifest, source, cases

    def test_missing_binding_is_recovered_only_before_any_observations_exist(self):
        manifest, source, cases = self.unbound_manifest()
        q.verify_manifest(manifest, source, cases, allow_unbound=True)
        with self.assertRaises(FileNotFoundError):
            q.verify_manifest(manifest, source, cases)
        result = {'artifacts': [], 'status': 'synthetic-finished'}
        budget = SimpleNamespace(writing=lambda *args: nullcontext())
        with patch.object(q, 'load_source', return_value=source), \
                patch.object(q, 'select_cases', return_value=cases), \
                patch.object(q.reference, 'validate_receipt'), \
                patch.object(q, 'check_dependencies', return_value=manifest['dependencies']), \
                patch.dict(sys.modules, {'run': SimpleNamespace(four_gpu_identity=Mock(return_value=manifest['hardware']))}), \
                patch.object(q.diagnostic, 'DiskBudget', return_value=budget), \
                patch.object(q, 'launch') as launch, \
                patch.object(q, 'collect', return_value=result), patch.object(q, 'show'), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(q.qualify(SimpleNamespace(gpus=4)), result)
        launch.assert_called_once()
        self.assertEqual(q.diagnostic.read(self.child / 'manifest-binding.json'),
                         {'sha256': q.inspector.digest(manifest)})
        self.assertEqual(q.diagnostic.read(self.child / 'experiment.json'), manifest)

    def test_unbound_manifest_with_any_other_output_is_not_resumed(self):
        manifest, source, cases = self.unbound_manifest()
        output = self.child / 'gpu-0.jsonl'
        output.write_text('{"saved":true}\n')
        with self.assertRaisesRegex(ValueError, 'already has other outputs'):
            q.verify_manifest(manifest, source, cases, allow_unbound=True)
        self.assertEqual(output.read_text(), '{"saved":true}\n')


class BudgetTests(Fixture):
    def test_child_reuses_original_budget_and_charges_old_outputs_shared_weights_and_new_results(self):
        runtime = self.root / '.runtime'
        weights = runtime / 'reference' / 'weights.gguf'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'w' * 32768)
        old = self.parent / 'closed-observations.jsonl'
        old.write_bytes(b'o' * 65536)
        with patch.object(q.reference, 'CACHE_ROOTS', ('.runtime/reference',)):
            original = q.diagnostic.DiskBudget(self.parent, create=True)
            budget_bytes = original.path.read_bytes()
            before = original.usage()
            self.assertGreaterEqual(before['reference_bytes'], 32768)
            self.assertGreaterEqual(before['controlled_and_cache_growth_bytes'], 32768 + 65536)
            resumed = q.diagnostic.DiskBudget(self.parent, create=False)
            with patch.object(q.diagnostic, 'ACTIVE_BUDGET', resumed):
                with (self.child / 'gpu-0.jsonl').open('ab') as stream:
                    q.diagnostic.append_observation(stream, {'raw': 'q' * 65536})
            after = resumed.usage()
            self.assertGreaterEqual(after['charged_bytes'] - before['charged_bytes'], 65536)
            self.assertEqual(original.path.read_bytes(), budget_bytes)
            self.assertEqual(old.read_bytes(), b'o' * 65536)


class SourceTests(Fixture):
    def source(self):
        code = self.root / 'old.py'
        code.write_text('frozen = True\n')
        self.stack.enter_context(patch.object(q.diagnostic, 'BASE_CODE', ['old.py']))
        self.stack.enter_context(patch.object(q.diagnostic, 'NEW_CODE', []))
        files = {}
        for name in ('registry', 'file', 'source_train', 'tokenizer', 'context_sidecar', 'context_provenance'):
            path = self.root / f'{name}.json'
            self.write(path, {'immutable': name})
            files[name] = q.diagnostic.record(path)
        panel = {key: entry for key, entry in files.items() if key != 'registry'}
        self.write(self.parent / 'disk-budget.json', {'original': True})
        manifest = {'train_panel': panel, 'source': {'assets': {'registry': files['registry']}},
                    'disk_budget': q.diagnostic.record(self.parent / 'disk-budget.json')}
        identity = {'code': {'git_commit': 'old-head', 'files': [q.diagnostic.record(code)]}}
        for name, value in (('identity.json', identity), ('experiment.json', manifest),
                             ('manifest-binding.json', {'sha256': q.inspector.digest(manifest)}),
                             ('train-panel.json', panel), ('stages/reference-transport.json', {'passed': False})):
            self.write(self.parent / name, value)
        result = {'status': 'TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED',
                  'reference_transport': {'observations': []},
                  'artifacts': [q.diagnostic.record(self.parent / name) for name in
                                ('experiment.json', 'manifest-binding.json', 'train-panel.json', 'disk-budget.json')]}
        self.write(self.parent / 'result.json', result)
        self.stack.enter_context(patch.object(q.inspector, 'load_report',
                                             return_value={'passed': False, 'counts': {'native_format': 0}}))
        return manifest, identity, result

    def test_source_checks_reuse_closed_files_without_writing_or_current_head_requirement(self):
        manifest, identity, _ = self.source()
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(q, 'code_record', side_effect=AssertionError('No current Git identity required')):
            source = q.load_source()
        self.assertEqual(source['manifest'], manifest)
        self.assertEqual(source['identity'], identity)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_changed_original_code_is_rejected_before_work(self):
        self.source()
        (self.root / 'old.py').write_text('frozen = False\n')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            q.load_source()

    def test_incomplete_original_code_inventory_is_rejected(self):
        _, identity, _ = self.source()
        identity['code']['files'] = []
        self.write(self.parent / 'identity.json', identity)
        with self.assertRaisesRegex(ValueError, 'code inventory is incomplete'):
            q.load_source()

    def test_parent_terminal_inventory_must_bind_panel_and_budget(self):
        _, _, result = self.source()
        for name in ('train-panel.json', 'disk-budget.json'):
            changed = deepcopy(result)
            changed['artifacts'] = [entry for entry in changed['artifacts'] if not entry['path'].endswith('/' + name)]
            self.write(self.parent / 'result.json', changed)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'terminal inventory'):
                q.load_source()

    def test_parent_panel_and_asset_tampering_are_rejected(self):
        manifest, _, _ = self.source()
        self.write(self.parent / 'train-panel.json', {'different': True})
        with self.assertRaisesRegex(ValueError, 'TRAIN panel metadata changed'):
            q.load_source()
        self.write(self.parent / 'train-panel.json', manifest['train_panel'])
        (self.root / 'context_sidecar.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            q.load_source()


class CompletedTests(Fixture):
    def completed(self, valid=30, semantic=1):
        cases = [self.case(index) for index in range(32)]
        for index, envelope in enumerate(cases):
            envelope['case']['passes'] = index < semantic
        manifest = {'case_ids': [row['case']['case_id'] for row in cases]}
        self.write(self.child / 'experiment.json', manifest)
        self.write(self.child / 'manifest-binding.json', {'sha256': q.inspector.digest(manifest)})
        observations = []
        for index, envelope in enumerate(cases):
            item = envelope['case']
            gpu = q.diagnostic.shard_for(item['case_id'])
            generation = self.generation(item, q.binding(manifest, gpu), valid=index < valid)
            observations.append(self.receipt(item, generation))
        for gpu in range(4):
            self.write_rows(self.child / f'gpu-{gpu}.jsonl',
                            [row for row in observations if q.diagnostic.shard_for(row['case_id']) == gpu])
        source = {'parent': self.parent, 'registry': self.parent / 'registry.json',
                  'manifest': {'train_panel': {}, 'reference': {}}}
        return cases, manifest, source

    def test_complete32_gates_reject_critical_and_unresolved_semantic_success(self):
        with patch.object(q.diagnostic, 'EvaluatorRPC', side_effect=lambda *args: self.rpc()):
            for valid, semantic in ((30, 1), (29, 1), (32, 0)):
                cases, manifest, source = self.completed(valid, semantic)
                report = q.collect(self.child, manifest, source, cases)
                self.assertEqual(report['passed'], valid >= 30 and semantic >= 1)
                self.assertEqual(report['metrics']['transport_valid']['planned'], 32)
                self.assertFalse(report['training_performed'])
                self.assertFalse(report['student_evaluations_repeated'])
                self.assertFalse(report['dev_evaluated'])
            for critical in ('detected', 'review_required'):
                cases, manifest, source = self.completed(32, 1)
                cases[0]['case']['critical_status'] = critical
                first_id = cases[0]['case']['case_id']
                path = self.child / f'gpu-{q.diagnostic.shard_for(first_id)}.jsonl'
                rows = list(q.diagnostic.rows(path))
                next(row for row in rows if row['case_id'] == first_id)['critical_status'] = critical
                self.write_rows(path, rows)
                report = q.collect(self.child, manifest, source, cases)
                self.assertFalse(report['passed'])
                self.assertEqual(report['metrics']['supported']['passed'], 0)

    def test_completed_rerun_has_no_model_gpu_download_or_new_generation(self):
        cases, manifest, source = self.completed()
        with patch.object(q.diagnostic, 'EvaluatorRPC', side_effect=lambda *args: self.rpc()):
            result = q.collect(self.child, manifest, source, cases)
            self.write(self.child / 'result.json', result)
            before = {path: path.read_bytes() for path in self.child.iterdir() if path.is_file()}
            with patch.object(q, 'load_source', return_value=source), \
                    patch.object(q, 'select_cases', return_value=cases), \
                    patch.object(q, 'verify_manifest'), \
                    patch.object(q, 'check_dependencies', side_effect=AssertionError('No runtime dependencies')), \
                    patch.object(q, 'launch', side_effect=AssertionError('No worker launch')), \
                    patch.object(q.transport, 'Runtime', side_effect=AssertionError('No model load')), \
                    patch.object(q.reference, 'validate_receipt', side_effect=AssertionError('No weight load')), \
                    patch.object(q.reference, 'prepare_reference', side_effect=AssertionError('No download')), \
                    patch.object(q.diagnostic, 'student_provider', side_effect=AssertionError('No student evaluation')), \
                    patch.object(q.diagnostic, 'dev_worker', side_effect=AssertionError('No DEV')), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(q.qualify(SimpleNamespace(gpus=4)), result)
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_duplicate_missing_and_extra_results_never_shrink_denominator(self):
        with patch.object(q.diagnostic, 'EvaluatorRPC', side_effect=lambda *args: self.rpc()):
            for mutation in ('duplicate', 'missing', 'unplanned'):
                cases, manifest, source = self.completed()
                path = self.child / 'gpu-0.jsonl'
                rows = list(q.diagnostic.rows(path))
                if mutation == 'duplicate':
                    rows.append(deepcopy(rows[0]))
                elif mutation == 'missing':
                    rows.pop()
                else:
                    rows[0]['case_id'] = 'unplanned'
                self.write_rows(path, rows)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    q.collect(self.child, manifest, source, cases)

    def test_wrong_shard_assignment_is_rejected_before_scoring(self):
        cases, manifest, source = self.completed()
        first, second = (self.child / f'gpu-{gpu}.jsonl' for gpu in (0, 1))
        left, right = first.read_bytes(), second.read_bytes()
        first.write_bytes(right)
        second.write_bytes(left)
        with patch.object(q.diagnostic, 'EvaluatorRPC', side_effect=AssertionError('No scoring of a wrong shard')):
            with self.assertRaisesRegex(ValueError, 'original assignment'):
                q.collect(self.child, manifest, source, cases)


class ExecutorIntegrationTests(unittest.TestCase):
    def test_synthetic_hermes_calls_match_native_delivery_and_do_not_repair_wrong_arguments(self):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(REPO / 'assets/tokenizer.json'))
        tokenizer.no_padding(); tokenizer.no_truncation()
        with (REPO / 'assets/train-full-v2.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                prompt = tokenizer.decode(row['input_ids'][:row['prompt_tokens']], skip_special_tokens=False)
                body = prompt[len(q.reference.START):-len(q.reference.END)].split(q.reference.SEPARATOR)[1]
                if not json.loads(body)['history']['exchanges'] and 'subject' in row['calls'][0]['arguments']:
                    break
            else:
                self.fail('No fixed empty-history TRAIN fixture is available')
        self.assertEqual(tokenizer.encode(prompt, add_special_tokens=False).ids,
                         row['input_ids'][:row['prompt_tokens']])
        with q.diagnostic.EvaluatorRPC('eval_tools/diagnose_data.py', REPO / 'assets/registry.json') as rpc:
            item = rpc.request({'action': 'prepare', 'source': 'common', 'row': row, 'prompt': prompt}, 'prepared')['case']
            baseline = rpc.request({'action': 'score', 'case': item,
                'generation': {'raw_text': item['exact_target'], 'complete': True}, 'compact': True}, 'score')['observation']

            def generate(calls):
                raw = Fixture.raw(item)
                raw['raw_text'] = ''.join('<tool_call>' + json.dumps(call) + '</tool_call>' for call in calls) + '<|im_end|>'
                return q.converted_generation(raw, item, 'synthetic-converter-fixture')

            actual = q.score(rpc, item, generate(row['calls']), 'synthetic-converter-fixture')
            self.assertTrue(actual['transport_valid'])
            self.assertTrue(actual['semantic_success'])
            self.assertEqual(actual['semantic_result'], baseline['semantic_result'])
            for key in ('after_sha256', 'after_semantic_sha256', 'delivered_sha256', 'delivered'):
                self.assertEqual(actual[key], baseline[key])
            wrong = deepcopy(row['calls'])
            wrong[0]['arguments']['subject'] = 'definitely-not-a-declared-subject'
            failed = q.score(rpc, item, generate(wrong), 'synthetic-converter-fixture')
            self.assertTrue(failed['transport_valid'])
            self.assertFalse(failed['semantic_success'])
            self.assertIn('definitely-not-a-declared-subject', failed['generation']['raw_text'])


if __name__ == '__main__':
    unittest.main()
