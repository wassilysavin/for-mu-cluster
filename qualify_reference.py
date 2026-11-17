"""Check only the original 32 TRAIN admission cases using Qwen Hermes calls."""

from private_runtime import configure_private_runtime

configure_private_runtime()

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import stat
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep

import diagnose as diagnostic
import inspect_reference as inspector
from ledger import code_record, execution_lease
import qwen_transport as transport
import reference_model as reference


ROOT = Path(__file__).resolve().parent
SOURCE_ID = 'train-fit-reference-v1'
REVISION = 'qwen-hermes-v1'
CHILD = f'runs/{SOURCE_ID}/revisions/{REVISION}'
CASES = 32
GPUS = 4
OUTPUT_ALLOWANCE = 16 * 1024**2
NEW_CODE = ['qualify_reference.py', 'qwen_transport.py', 'inspect_reference.py']
CONVERSION_FIELDS = ('native_text', 'transport_valid', 'transport_error', 'complete', 'call_count')


def output_directory():
    path = ROOT
    for part in Path(CHILD).parts:
        path = path / part
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError('Qualification directories must be real private directories')
    if path.exists():
        for parent, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                entry = Path(parent) / name
                try:
                    info = entry.lstat()
                except FileNotFoundError:
                    continue
                if entry.is_symlink() or (name in files and not stat.S_ISREG(info.st_mode)):
                    raise ValueError('Qualification outputs must not redirect to another file')
    return path


def writable_file(path):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
        raise ValueError('Writable qualification file must not alias another file')
    return path


def load_source():
    report = inspector.load_report(SOURCE_ID)
    parent = inspector.inside(ROOT / 'runs' / SOURCE_ID)
    manifest = diagnostic.read(parent / 'experiment.json')
    identity = diagnostic.read(parent / 'identity.json')
    result = diagnostic.read(parent / 'result.json')
    if report['passed'] or result['status'] != 'TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED':
        raise ValueError('This revision requires the original completed, unqualified reference diagnostic')
    required = set(diagnostic.BASE_CODE + diagnostic.NEW_CODE)
    original_code = identity['code']['files']
    if not required.issubset({item['path'] for item in original_code}):
        raise ValueError('Original diagnostic code inventory is incomplete')
    for entry in original_code:
        diagnostic.verify(entry)
    if manifest['train_panel'] != diagnostic.read(parent / 'train-panel.json'):
        raise ValueError('Original TRAIN panel metadata changed')
    diagnostic.verify(manifest['disk_budget'])
    registry = diagnostic.verify(manifest['source']['assets']['registry'])
    evidence = [diagnostic.record(parent / name) for name in
                ('identity.json', 'experiment.json', 'manifest-binding.json', 'result.json',
                 'train-panel.json', 'disk-budget.json', 'stages/reference-transport.json')]
    for name in ('experiment.json', 'manifest-binding.json', 'train-panel.json', 'disk-budget.json'):
        if diagnostic.record(parent / name) not in result['artifacts']:
            raise ValueError(f'Original terminal inventory does not bind {name}')
    evidence += result['reference_transport']['observations']
    evidence += [manifest['source']['assets']['registry']]
    for key in ('file', 'source_train', 'tokenizer', 'context_sidecar', 'context_provenance'):
        entry = manifest['train_panel'][key]
        diagnostic.verify(entry)
        evidence.append(entry)
    return {'parent': parent, 'manifest': manifest, 'identity': identity, 'registry': registry,
            'evidence': evidence, 'counts': report['counts']}


def select_cases(panel):
    from tokenizers import Tokenizer

    planned = panel['qualification_ids']
    if len(planned) != CASES or len(set(planned)) != CASES:
        raise ValueError('Exactly the original 32 distinct qualification IDs are required')
    original_rows = list(diagnostic.rows(diagnostic.verify(panel['source_train'])))
    original = {row['case_id']: row for row in original_rows}
    if len(original) != len(original_rows):
        raise ValueError('Duplicate case in original TRAIN')
    tokenizer = Tokenizer.from_file(str(diagnostic.verify(panel['tokenizer'])))
    tokenizer.no_padding(); tokenizer.no_truncation()
    selected = {}
    for envelope in diagnostic.rows(diagnostic.verify(panel['file'])):
        item = envelope['case']
        if item['case_id'] not in planned:
            continue
        if item['case_id'] in selected or envelope.pop('storage', None) != 'frozen-train-prefix-v1':
            raise ValueError('Duplicate or changed prepared qualification case')
        if 'prompt' in item:
            raise ValueError('Expected the immutable tokenized TRAIN prefix')
        row = original[item['case_id']]
        prefix = row['input_ids'][:row['prompt_tokens']]
        prompt = tokenizer.decode(prefix, skip_special_tokens=False)
        if (sha256(prompt.encode()).hexdigest() != item['prompt_sha256']
                or tokenizer.encode(prompt, add_special_tokens=False).ids != prefix
                or item['gold_calls'] != row['calls']
                or tokenizer.decode(row['input_ids'][row['prompt_tokens']:], skip_special_tokens=False) != item['exact_target']):
            raise ValueError('Qualification prompt or target differs from original TRAIN')
        item['prompt'] = prompt
        transport.adapt_prompt(prompt)
        selected[item['case_id']] = envelope
    if set(selected) != set(planned):
        raise ValueError('Missing original qualification cases')
    return [selected[case_id] for case_id in planned]


def code_identity(source):
    names = [entry['path'] for entry in source['identity']['code']['files']] + NEW_CODE
    return code_record(ROOT, sorted(set(names)))


def binding(manifest, gpu):
    return inspector.digest({'manifest': inspector.digest(manifest), 'gpu': gpu,
                             'phase': 'qualification', 'revision': REVISION})


def converted_generation(record, item, worker_binding):
    converted = transport.convert_generation(record)
    return {**record, **converted, 'reference_raw_text': record['raw_text'],
            'raw_text': converted['native_text'] if converted['complete'] else record['raw_text'],
            'raw_ids': record['raw_token_ids'], 'provider_kind': 'qwen3-hermes',
            'diagnostic_worker_sha256': worker_binding,
            'transport_recipe_sha256': inspector.digest(transport.RECIPE)}


def verify_generation(generation, item, worker_binding):
    if (generation.get('diagnostic_worker_sha256') != worker_binding
            or generation.get('transport_recipe_sha256') != inspector.digest(transport.RECIPE)
            or generation.get('input_sha256') != item['prompt_sha256']
            or generation.get('provider_kind') != 'qwen3-hermes'
            or generation.get('max_new_tokens') != transport.RECIPE['max_new_tokens']):
        raise ValueError('Saved generation prompt or worker identity changed')
    if generation.get('error'):
        if (generation['error'] != 'context_overflow' or generation.get('raw_text') != ''
                or generation.get('raw_ids') != [] or generation.get('complete') is not False
                or generation['context_tokens'] != reference.CONTEXT_TOKENS
                or generation['input_tokens'] + generation['max_new_tokens'] <= reference.CONTEXT_TOKENS):
            raise ValueError('Invalid saved context-overflow receipt')
        return
    original = {**generation, 'raw_text': generation['reference_raw_text'],
                'raw_token_ids': generation['raw_ids']}
    converted = transport.convert_generation(original)
    if (any(generation.get(key) != converted[key] for key in CONVERSION_FIELDS)
            or generation['raw_text'] != (converted['native_text'] if converted['complete'] else original['raw_text'])
            or generation.get('raw_token_ids') != generation['raw_ids']
            or generation.get('generated_tokens') != len(generation['raw_ids'])
            or any(type(token) is not int or token < 0 for token in generation['raw_ids'])
            or len(generation['raw_ids']) > generation['max_new_tokens']
            or not isinstance(generation.get('rendered_prompt_sha256'), str)
            or len(generation['rendered_prompt_sha256']) != 64):
        raise ValueError('Saved Hermes conversion or raw token record changed')


def score(rpc, item, generation, worker_binding):
    verify_generation(generation, item, worker_binding)
    return rpc.request({'action': 'score', 'case': item, 'generation': generation,
                        'compact': True}, 'score')['observation']


def run_shard(cases, path, rpc, generate, worker_binding, saved=None):
    writable_file(path)
    saved = diagnostic.recover_jsonl(path) if saved is None else saved
    if len(saved) > len(cases):
        raise ValueError('Saved shard exceeds planned qualification cases')
    with path.open('ab') as stream:
        for index, envelope in enumerate(cases):
            item = envelope['case']
            previous = saved[index] if index < len(saved) else None
            if previous and (previous['case_id'] != item['case_id'] or previous['case_sha256'] != item['case_sha256']):
                raise ValueError('Saved shard case order or identity changed')
            generation = previous['generation'] if previous else generate(item)
            observation = score(rpc, item, generation, worker_binding)
            if previous:
                if diagnostic.canonical(previous) != diagnostic.canonical(observation):
                    raise ValueError('Saved qualification observation no longer reproduces')
            else:
                diagnostic.append_observation(stream, observation)
                print(f'[Qwen GPU {diagnostic.shard_for(item["case_id"])}] qualification {index + 1}/{len(cases)}', flush=True)


def verify_manifest(manifest, source, cases, *, allow_unbound=False):
    if (manifest.get('schema') != 1 or manifest.get('revision') != REVISION
            or manifest.get('source_evidence') != source['evidence']
            or manifest.get('reference') != source['manifest']['reference']
            or manifest.get('recipe') != transport.RECIPE
            or manifest.get('case_ids') != [item['case']['case_id'] for item in cases]
            or manifest.get('case_hashes') != [item['case']['case_sha256'] for item in cases]
            or manifest.get('policy') != {'cases': CASES, 'gpus': GPUS, 'format_minimum': 30,
                                         'supported_minimum': 1, 'training': False, 'dev': False}):
        raise ValueError('Qualification manifest changed or belongs to another source')
    current = code_identity(source)
    if current['files'] != manifest['code']['files']:
        raise ValueError('Qualification code changed; keep the frozen run intact')
    child = output_directory()
    if allow_unbound and not (child / 'manifest-binding.json').exists():
        if {path.name for path in child.iterdir()} == {'experiment.json'}:
            return
        raise ValueError('Unbound qualification manifest already has other outputs')
    if diagnostic.read(child / 'manifest-binding.json') != {'sha256': inspector.digest(manifest)}:
        raise ValueError('Qualification manifest binding changed')


def collect(child, manifest, source, cases):
    paths = [child / f'gpu-{gpu}.jsonl' for gpu in range(GPUS)]
    for gpu, path in enumerate(paths):
        planned = [case_id for case_id in manifest['case_ids'] if diagnostic.shard_for(case_id) == gpu]
        if [row['case_id'] for row in diagnostic.rows(path)] != planned:
            raise ValueError('Saved GPU shard differs from its original assignment')
    all_rows = diagnostic.merged_rows(paths, manifest['case_ids'])
    with diagnostic.NodeRPC('diagnose-data.mjs', source['registry']) as rpc:
        for envelope, row in zip(cases, all_rows):
            item = envelope['case']
            if row['case_sha256'] != item['case_sha256']:
                raise ValueError('Qualification observation case hash differs')
            expected = score(rpc, item, row['generation'], binding(manifest, diagnostic.shard_for(item['case_id'])))
            if diagnostic.canonical(expected) != diagnostic.canonical(row):
                raise ValueError('Qualification score no longer reproduces')
    counts = diagnostic.metrics(all_rows)
    if counts['transport_valid']['planned'] != CASES or counts['semantic_ineligible']:
        raise ValueError('Qualification denominator or semantic eligibility changed')
    passed = counts['transport_valid']['passed'] >= 30 and counts['supported']['passed'] >= 1
    examples = []
    if not passed:
        failures = sorted((row for row in all_rows if not row['success']),
                          key=lambda row: (row['transport_valid'], row['case_id']))
        for row in failures[:2]:
            raw = row['generation'].get('reference_raw_text', '')
            examples.append({'case_id': row['case_id'], 'error': row.get('error_kind'),
                             'raw_prefix': raw[:600], 'raw_suffix': raw[-200:] if len(raw) > 600 else ''})
    return {'status': 'QUALIFIED' if passed else 'NOT_QUALIFIED', 'passed': passed,
            'manifest_sha256': inspector.digest(manifest), 'cases': CASES,
            'metrics': {key: counts[key] for key in ('transport_valid', 'supported', 'critical_detected', 'unresolved')},
            'generation_stops': dict(sorted(Counter(row['generation'].get('finish_reason', 'context_overflow') for row in all_rows).items())),
            'transport_errors': dict(sorted(Counter(row['generation'].get('transport_error') or row['generation'].get('error') or 'none' for row in all_rows).items())),
            'failure_examples': examples,
            'artifacts': [diagnostic.record(path) for path in paths + [child / 'experiment.json', child / 'manifest-binding.json']],
            'training_performed': False, 'student_evaluations_repeated': False, 'dev_evaluated': False,
            'reference_knowledge_established': False, 'release_ready': False}


def show(result, child):
    print(f'Qwen Hermes qualification: {result["status"]}')
    for key, label in (('transport_valid', 'Valid tool-call batches'), ('supported', 'Complete supported answers')):
        count = result['metrics'][key]
        print(f'{label}: {count["passed"]}/{count["planned"]}')
    print('Generation stops:', diagnostic.canonical(result['generation_stops']))
    print('Transport errors:', diagnostic.canonical(result['transport_errors']))
    for example in result['failure_examples']:
        print('Failed example:', diagnostic.canonical(example))
    print('Only 32 TRAIN cases; no training, student evaluations or DEV were run.')
    print('Report:', child / 'result.json')


def worker(args):
    import run as runner

    if args.lease_fd is None or args.parent_fd is None:
        raise ValueError('Workers require the inherited execution lease and parent pipe')
    runner.verify_worker_lease(ROOT / 'runs', args.lease_fd)
    def parent_watch():
        os.read(args.parent_fd, 1)
        os._exit(72)
    Thread(target=parent_watch, daemon=True).start()
    source = load_source()
    diagnostic.ACTIVE_BUDGET = diagnostic.DiskBudget(source['parent'], create=False)
    child = output_directory()
    manifest = diagnostic.read(child / 'experiment.json')
    cases = select_cases(source['manifest']['train_panel'])
    verify_manifest(manifest, source, cases)
    cases = [item for item in cases if diagnostic.shard_for(item['case']['case_id']) == args.worker]
    path = writable_file(child / f'gpu-{args.worker}.jsonl')
    saved = diagnostic.recover_jsonl(path)
    worker_binding = binding(manifest, args.worker)
    with diagnostic.NodeRPC('diagnose-data.mjs', source['registry']) as rpc:
        if len(saved) > len(cases):
            raise ValueError('Saved qualification shard exceeds the plan')
        for item, row in zip(cases, saved):
            if row != score(rpc, item['case'], row['generation'], worker_binding):
                raise ValueError('Saved qualification prefix no longer reproduces')
        runtime = None
        if len(saved) < len(cases):
            if check_dependencies() != manifest['dependencies']:
                raise ValueError('Worker dependencies differ from the frozen qualification')
            if runner.configure_parallel_numerics() != manifest['hardware']['numerics']:
                raise ValueError('Worker numerical policy differs from the coordinator')
            os.environ['CUDA_CACHE_DISABLE'] = '1'
            runtime = transport.Runtime(manifest['reference'], f'cuda:{args.worker}')
        diagnostic.publish_snapshot(writable_file(child / f'ready-{args.worker}.json'),
                                    {'pid': os.getpid(), 'attempt': args.attempt})
        def generate(item):
            if runtime is None:
                raise RuntimeError('Completed shard must not generate again')
            try:
                raw = runtime.generate_record(item['prompt'], max_tokens=transport.RECIPE['max_new_tokens'])
                return converted_generation(raw, item, worker_binding)
            except reference.ReferenceContextOverflow as error:
                return {'error': 'context_overflow', 'raw_text': '', 'raw_ids': [], 'complete': False,
                        'provider_kind': 'qwen3-hermes', 'input_tokens': error.input_tokens,
                        'max_new_tokens': error.max_new_tokens, 'context_tokens': error.context_tokens,
                        'input_sha256': item['prompt_sha256'], 'diagnostic_worker_sha256': worker_binding,
                        'transport_recipe_sha256': inspector.digest(transport.RECIPE)}
        run_shard(cases, path, rpc, generate, worker_binding, saved)


def launch(child, lease):
    read_fd, write_fd = os.pipe()
    attempt = f'{os.getpid()}-{monotonic()}'
    processes = []
    try:
        for gpu in range(GPUS):
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'qualify_reference.py'),
                '--worker', str(gpu), '--lease-fd', str(lease), '--parent-fd', str(read_fd),
                '--attempt', attempt], pass_fds=(lease, read_fd))
            processes.append(process)
            ready = child / f'ready-{gpu}.json'
            while not ready.exists() or diagnostic.read(ready) != {'pid': process.pid, 'attempt': attempt}:
                if process.poll() is not None or any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('Qualification worker failed during startup; saved cases remain resumable')
                sleep(.25)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Qualification worker failed; repeat the same command to resume')
            sleep(.5)
        if any(p.returncode for p in processes):
            raise RuntimeError('Qualification worker failed; no completed report was published')
    finally:
        os.close(write_fd); os.close(read_fd)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def check_dependencies():
    expected = {'transformers': '4.57.3', 'huggingface-hub': '0.36.2', 'tokenizers': '0.22.2',
                **reference.REFERENCE_DEPENDENCIES}
    found = {name: version(name) for name in expected}
    if found != expected:
        raise RuntimeError('Use the already installed pinned reference dependencies; no automatic installation occurs')
    return found


def qualify(args):
    source = load_source()
    cases = select_cases(source['manifest']['train_panel'])
    child = output_directory()
    with execution_lease(ROOT / 'runs') as lease:
        manifest = diagnostic.read(child / 'experiment.json') if (child / 'experiment.json').exists() else None
        if manifest is not None:
            verify_manifest(manifest, source, cases, allow_unbound=not (child / 'result.json').exists())
        if (child / 'result.json').exists():
            result = diagnostic.read(child / 'result.json')
            for entry in result['artifacts']:
                diagnostic.verify(entry)
            if manifest is None or result != collect(child, manifest, source, cases):
                raise ValueError('Completed qualification result no longer reproduces')
            show(result, child)
            return result
        reference.validate_receipt(source['manifest']['reference'], full_hash=True)
        dependencies = check_dependencies()
        import run as runner
        hardware = runner.four_gpu_identity()
        diagnostic.ACTIVE_BUDGET = diagnostic.DiskBudget(source['parent'], create=False)
        with diagnostic.ACTIVE_BUDGET.writing(OUTPUT_ALLOWANCE):
            pass
        child = diagnostic.directory(CHILD)
        if manifest is None:
            manifest = {'schema': 1, 'revision': REVISION, 'source_evidence': source['evidence'],
                        'reference': source['manifest']['reference'], 'recipe': transport.RECIPE,
                        'case_ids': [item['case']['case_id'] for item in cases],
                        'case_hashes': [item['case']['case_sha256'] for item in cases],
                        'code': code_identity(source), 'hardware': hardware, 'dependencies': dependencies,
                        'policy': {'cases': CASES, 'gpus': GPUS, 'format_minimum': 30,
                                   'supported_minimum': 1, 'training': False, 'dev': False}}
            diagnostic.immutable(child / 'experiment.json', manifest)
        elif manifest['hardware'] != hardware or manifest['dependencies'] != dependencies:
            raise ValueError('Qualification hardware or dependencies changed on resume')
        diagnostic.immutable(child / 'manifest-binding.json', {'sha256': inspector.digest(manifest)})
        print('Qwen Hermes: the original 32 TRAIN cases on GPUs 0, 1, 2, 3; existing weights only.', flush=True)
        launch(child, lease)
        for entry in source['evidence']:
            diagnostic.verify(entry)
        result = collect(child, manifest, source, cases)
        diagnostic.publish_json(writable_file(child / 'result.json'), result)
        show(result, child)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=[4], default=4)
    parser.add_argument('--worker', type=int, choices=range(GPUS), help=argparse.SUPPRESS)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--parent-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args) if args.worker is not None else qualify(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(f'Qwen qualification error: {error}')
