"""Measure frozen TRAIN fit and a pinned reference after the history experiment."""

from private_runtime import configure_private_runtime, private_directory

configure_private_runtime()

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from shutil import disk_usage
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep

import history
from ledger import SAFE_ID, code_record, execution_lease, file_record, publish_json
import reference_model

ROOT = Path(__file__).resolve().parent
RUN_ID = 'train-fit-reference-v1'
RESERVE = 512 * 1024**2
OUTPUT_ALLOWANCE = 3 * 1024**3
SEED = 1729
SHARDS = 4
PILOT_CASES = 32
PILOT_TRANSPORT_MINIMUM = 30
BASE_CODE = ['model.py', 'ledger.py', 'private_runtime.py', 'cluster', 'history.py', 'history_prepare.py',
             'run.py', 'requirements.txt', 'experiments.json', 'experiments-history.json']
NEW_CODE = ['diagnose.py', 'diagnose-data.mjs', 'diagnose-dev.mjs', 'reference_model.py']


def read(path):
    return json.loads(history.contained(path, ROOT).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def rows(path):
    with history.contained(path, ROOT).open() as stream:
        for line in stream:
            yield json.loads(line)


def record(path):
    return file_record(history.contained(path, ROOT), relative_to=ROOT)


def verify(entry):
    path = history.contained(ROOT / entry['path'], ROOT)
    if record(path) != entry:
        raise ValueError(f"Frozen artifact changed: {entry['path']}")
    return path


def directory(relative):
    return private_directory(ROOT, relative)


def require_space(path, extra=0):
    if disk_usage(path).free < RESERVE + extra:
        raise RuntimeError(f'Need {(RESERVE + extra) / 1e9:.2f} GB free inside this checkout; no external folders are used')


def latest_source(runs, source_id):
    if not SAFE_ID.fullmatch(source_id):
        raise ValueError('Invalid source run ID')
    parent = history.contained(runs / source_id, runs)
    candidates = [parent / 'revisions/rope-alignment-v1/recovery/checkpoint-io-v1',
                  parent / 'revisions/rope-alignment-v1', parent]
    for candidate in candidates:
        candidate = history.contained(candidate, runs)
        if candidate.exists() and any(candidate.iterdir()):
            experiment = history.contained(candidate / 'experiment.json', ROOT)
            result = history.contained(candidate / 'result.json', ROOT)
            if not experiment.is_file() or not result.is_file():
                raise RuntimeError(f'Current history experiment has no terminal result: {candidate}. Finish or recover it first; no reference model was downloaded.')
            return candidate
    raise FileNotFoundError(f'No completed history experiment under {parent}')


def source_context(runs, source_id):
    import run as runner
    source = latest_source(runs, source_id)
    manifest, result = read(source / 'experiment.json'), read(source / 'result.json')
    binding = history.digest(manifest)
    if result.get('identity_sha256') != binding:
        raise ValueError('Source result and frozen experiment identities differ')
    if result.get('status') not in {'confirmed_diagnostic_gain', 'exploratory_gain_second_seed_not_run',
                                   'diagnostic_gain', 'inconclusive', 'not_promoted', 'budget_stopped'}:
        raise RuntimeError('The source does not have a recognized terminal comparison status')
    pairs = [pair for pair in result.get('pairs', []) if pair.get('seed') == SEED]
    if len(pairs) != 1 or not result.get('baseline') or not all(pairs[0].get(arm) for arm in ('control', 'treatment')):
        raise RuntimeError('Need the completed baseline and both complete seed-1729 arms; a partial or budget-stopped pair is not silently substituted')
    for entry in list(manifest['assets'].values()) + manifest['code']['files']:
        verify(entry)
    parity = read(source / 'stages/official-parity.json')
    if parity.get('identity_sha256') != binding or parity.get('result', {}).get('passed') is not True:
        raise ValueError('The original strict numerical gate has no authenticated PASS receipt')
    prepared_receipt = read(source / 'stages/prepare.json')
    if prepared_receipt.get('identity_sha256') != binding:
        raise ValueError('Prepared TRAIN belongs to another experiment')
    prepared = prepared_receipt['result']
    fingerprint = prepared['collection_receipt']['fingerprint']
    for key, asset in (('checkpoint_sha256', 'initial_checkpoint'), ('registry_sha256', 'registry'),
                       ('tokenizer_sha256', 'tokenizer'), ('specs_sha256', 'history_train_specs')):
        if fingerprint.get(key) != manifest['assets'][asset]['sha256']:
            raise ValueError(f'Collected TRAIN used a different {asset}')
    train, schedules = history.validate_prepared(prepared, source, runner)
    identities = read(source / 'identities.json')
    origin = manifest['origin']
    for entry in origin.values():
        if isinstance(entry, dict) and {'path', 'bytes', 'sha256'} <= entry.keys():
            verify(entry)
    selected = {'baseline-7205': result['baseline'], **{
        f'{arm}-{SEED}': pairs[0][arm] for arm in ('control', 'treatment')}}
    evidence = [record(source / name) for name in ('experiment.json', 'result.json',
                'stages/official-parity.json', 'stages/prepare.json', 'identities.json')]
    for name, evaluated in selected.items():
        for key in ('checkpoint', 'observations', 'timings'):
            verify(evaluated[key])
            evidence.append(evaluated[key])
        history.validate_score(evaluated['score'], ROOT / evaluated['observations']['path'], manifest['config'])
        if sum(row['seconds'] for row in history.timing_records(ROOT / evaluated['timings']['path'])) != evaluated['seconds']:
            raise ValueError('Source evaluation timing receipt changed')
        if evaluated['score'].get('infrastructure_failures'):
            raise RuntimeError(f'{name} has infrastructure failures; resolve the source evaluation first')
        expected_step = 7205 if name == 'baseline-7205' else 7991
        metadata = history.checkpoint_header(ROOT / evaluated['checkpoint']['path'])['__metadata__']
        step = int(metadata['step'])
        lineage = json.loads(metadata.get('lineage', 'null'))
        if step != expected_step:
            raise ValueError(f'Unexpected {name} checkpoint step {step}')
        if name == 'baseline-7205':
            if evaluated['checkpoint'] != origin['checkpoint'] or lineage != origin['lineage']:
                raise ValueError('Baseline is not the authenticated original 7205')
        else:
            identity = identities[name]
            if (identity['experiment_id'] != name or identity['initial_checkpoint'] != origin['checkpoint']
                    or identity['prepared_sha256'] != history.digest(prepared)
                    or identity['train']['schedule_sha256'] != schedules[name].sha256
                    or identity['train']['arm'] != name
                    or lineage != {'identity_sha256': history.digest(identity),
                                   'initial_checkpoint_sha256': origin['checkpoint']['sha256']}):
                raise ValueError('Completed arm checkpoint lineage or TRAIN membership differs')
    evidence.append(record(train[0]))
    return {'directory': source, 'manifest': manifest, 'result': result, 'prepared': prepared,
            'train': train[0], 'students': selected, 'evidence': evidence}


class NodeRPC:
    def __init__(self, script, registry):
        self.process = subprocess.Popen(['node', str(ROOT / script), str(registry)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)

    def __enter__(self):
        return self

    def request(self, value, expected):
        self.process.stdin.write(canonical(value) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Diagnostic evaluator disconnected (exit {self.process.poll()})')
        result = json.loads(line)
        if result.get('kind') != expected:
            raise ValueError(f'Unexpected diagnostic response: {result.get("kind")}')
        return result

    def __exit__(self, kind, value, trace):
        self.process.stdin.close()
        if kind is not None:
            self.process.terminate()
        status = self.process.wait()
        self.process.stdout.close()
        if kind is None and status:
            raise RuntimeError(f'Diagnostic evaluator failed: {status}')


def qualification_ids(cases):
    groups = defaultdict(list)
    for envelope in cases:
        item = envelope['case']
        if item['semantic_eligible']:
            groups[item['group']].append(item['case_id'])
    for group in groups.values():
        group.sort(key=lambda value: sha256(value.encode()).hexdigest())
    longest = max(cases, key=lambda row: (row['input_token_count'], row['case']['case_id']))
    selected = [longest['case']['case_id']]
    for group in groups.values():
        if selected[0] in group: group.remove(selected[0])
    while len(selected) < PILOT_CASES:
        before = len(selected)
        for name in sorted(groups):
            if groups[name] and len(selected) < PILOT_CASES:
                selected.append(groups[name].pop(0))
        if len(selected) == before:
            raise ValueError('Not enough eligible TRAIN cases for the frozen transport admission')
    return selected


def validate_contexts():
    manifest = read(ROOT / 'assets/diagnostic-train-contexts.json')
    path = ROOT / 'assets/diagnostic-train-contexts.jsonl'
    actual = record(path)
    if any(actual[key] != manifest['contexts'][key] for key in ('path', 'bytes', 'sha256')):
        raise ValueError('Restored TRAIN contexts changed after verification')
    original = read(ROOT / 'experiments.json')['assets']
    replay = read(ROOT / 'experiments-history.json')['assets']['history_replay']
    if (manifest['split'] != 'train' or manifest['source_dataset']['sha256'] != replay['sha256']
            or manifest['tokenizer']['sha256'] != original['tokenizer']['sha256']
            or manifest['registry_sha256'] != read(ROOT / original['registry']['path'])['fingerprint']):
        raise ValueError('Restored TRAIN belongs to different data, tokenizer, or domain contracts')
    original_file = record(ROOT / replay['path'])
    if (manifest['source_dataset']['bytes'] != original_file['bytes'] or manifest['rows_in_source'] != 2977
            or manifest['restored_history_rows'] != 720 or manifest['contexts']['rows'] != 720):
        raise ValueError('Restored TRAIN provenance counts differ')
    contexts = list(rows(path))
    if len(contexts) != 720 or len({row['case_id'] for row in contexts}) != 720:
        raise ValueError('The sidecar must preserve all 720 distinct historical TRAIN contexts')
    return manifest, {row['case_id']: row for row in contexts}


def panel_binding(source):
    return history.digest({'train': record(source['train']), 'contexts': record(ROOT / 'assets/diagnostic-train-contexts.jsonl'),
                           'prepared': history.digest(source['prepared']),
                           'preparer': record(ROOT / 'diagnose-data.mjs')})


def prepare_panel(source, output, registry):
    _, contexts = validate_contexts()
    context_path = ROOT / 'assets/diagnostic-train-contexts.jsonl'
    collection = source['directory'] / 'collection'
    receipt = source['prepared']['collection_receipt']
    histories = {}
    for arm in ('control', 'treatment'):
        for raw in rows(collection / receipt[arm]['path']):
            histories[f'{arm}:{raw["case_id"]}'] = raw
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / source['manifest']['assets']['tokenizer']['path']))
    tokenizer.no_padding(); tokenizer.no_truncation()
    cases = []
    with NodeRPC('diagnose-data.mjs', registry) as rpc:
        for row in rows(source['train']):
            case_id = row['case_id']
            prompt = tokenizer.decode(row['input_ids'][:row['prompt_tokens']], skip_special_tokens=False)
            target = tokenizer.decode(row['input_ids'][row['prompt_tokens']:], skip_special_tokens=False)
            if tokenizer.encode(prompt, add_special_tokens=False).ids != row['input_ids'][:row['prompt_tokens']]:
                raise ValueError(f'Frozen prompt does not roundtrip: {case_id}')
            raw = dict(histories.get(case_id, row), case_id=case_id)
            request = {'action': 'prepare', 'source': 'history' if case_id in histories else 'common',
                       'row': raw, 'prompt': prompt}
            if case_id in contexts:
                request['before'] = contexts[case_id]['before']
            item = rpc.request(request, 'prepared')['case']
            if item['exact_target'] != target or not item['semantic_eligible']:
                raise ValueError(f'TRAIN context/target cannot be fully authenticated: {case_id}')
            exposure = {name: len(row['positions'].get(name, [])) for name in source['students'] if name != 'baseline-7205'}
            exposure['baseline-7205'] = 1 if item['source'] == 'common' else 0
            cases.append({'case': item, 'exposure': exposure, 'input_token_count': row['prompt_tokens'],
                          'baseline_exposure_count_is_membership_only': True})
    if len(cases) != 2532 or len(histories) != 960:
        raise ValueError('Frozen TRAIN union must contain 1572 common and 480 cases per arm')
    for name in ('control-1729', 'treatment-1729'):
        if sum(x['exposure'][name] > 0 for x in cases) != 2052 or sum(x['exposure'][name] for x in cases) != 3144:
            raise ValueError('TRAIN membership or exposure differs from the completed schedule')
    info = {'schema': 1, 'binding': panel_binding(source), 'cases': len(cases), 'all_semantic_eligible': True,
            'strata': dict(Counter(x['case']['source'] for x in cases)),
            'qualification_ids': qualification_ids(cases), 'context_sidecar': record(context_path),
            'context_provenance': record(ROOT / 'assets/diagnostic-train-contexts.json'),
            'baseline_fit_scope': '1572 replay members of earlier full-v2 TRAIN, not all 2977 original cases',
            'reference_is_not_trained_on_this_panel': True}
    path = output / 'train-panel.jsonl'
    with TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / path.name
        with staged.open('w') as stream:
            for envelope in cases:
                stream.write(canonical(envelope) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            if file_record(staged)['sha256'] != record(path)['sha256']:
                raise ValueError('Interrupted prepared panel differs from deterministic restoration')
        else:
            os.rename(staged, path)
    info['file'] = record(path)
    publish_json(output / 'train-panel.json', info)
    return info


def validate_panel(info, source):
    if info.get('binding') != panel_binding(source):
        raise ValueError('Prepared diagnostic panel belongs to different TRAIN or preparation code')
    cases = list(rows(verify(info['file'])))
    original = {row['case_id']: row for row in rows(source['train'])}
    if len(cases) != 2532 or {x['case']['case_id'] for x in cases} != set(original):
        raise ValueError('Diagnostic panel changed its full TRAIN denominator')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / source['manifest']['assets']['tokenizer']['path']))
    tokenizer.no_padding(); tokenizer.no_truncation()
    for envelope in cases:
        item = envelope['case']; row = original[item['case_id']]
        if (not item['semantic_eligible'] or item['gold_calls'] != row['calls']
                or sha256(item['prompt'].encode()).hexdigest() != item['prompt_sha256']
                or tokenizer.encode(item['prompt'], add_special_tokens=False).ids != row['input_ids'][:row['prompt_tokens']]
                or tokenizer.decode(row['input_ids'][row['prompt_tokens']:], skip_special_tokens=False) != item['exact_target']):
            raise ValueError('Diagnostic semantic case or exact frozen token sequence changed')
        expected_source = 'history' if item['case_id'].startswith(('control:', 'treatment:')) else 'common'
        if item['source'] != expected_source: raise ValueError('TRAIN source membership changed')
        expected = {name: len(row['positions'].get(name, [])) for name in source['students'] if name != 'baseline-7205'}
        expected['baseline-7205'] = int(item['source'] == 'common')
        if envelope['exposure'] != expected or envelope['input_token_count'] != row['prompt_tokens']:
            raise ValueError('Diagnostic TRAIN exposure changed')
    if info['qualification_ids'] != qualification_ids(cases):
        raise ValueError('Reference admission cases changed')
    return cases


def shard_for(case_id):
    return int(sha256(case_id.encode()).hexdigest()[:8], 16) % SHARDS


def recover_jsonl(path):
    path = history.contained(path, ROOT)
    if not path.exists():
        return []
    with path.open('rb+') as stream:
        data = stream.read()
        if data and not data.endswith(b'\n'):
            boundary = data.rfind(b'\n') + 1
            try:
                json.loads(data[boundary:])
                stream.seek(0, 2); stream.write(b'\n'); data += b'\n'
            except (ValueError, UnicodeDecodeError):
                stream.seek(boundary); stream.truncate(); data = data[:boundary]
            stream.flush(); os.fsync(stream.fileno())
    return [json.loads(line) for line in data.splitlines()]


def append_observation(stream, value):
    encoded = (canonical(value) + '\n').encode()
    require_space(Path(stream.name).parent, len(encoded))
    stream.write(encoded); stream.flush(); os.fsync(stream.fileno())


def student_provider(checkpoint, assets, gpu):
    import torch
    import run as runner
    from model import Runtime
    from tokenizers import Tokenizer
    runner.configure_parallel_numerics()
    torch.cuda.set_device(gpu)
    runtime = Runtime(ROOT / assets['base']['path'], checkpoint, device=f'cuda:{gpu}', expected_batch_size=4)
    tokenizer = Tokenizer.from_file(str(ROOT / assets['tokenizer']['path']))
    tokenizer.no_padding(); tokenizer.no_truncation()
    def generate(prompt):
        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(tokens) + 512 > 32768:
            return {'provider_kind': 'functiongemma', 'raw_text': '', 'raw_ids': [],
                    'error': 'context_overflow', 'complete': False}
        started = monotonic()
        raw = runtime.generate(tokens, max_new_tokens=512, stop_ids={50})
        return {'provider_kind': 'functiongemma', 'raw_text': tokenizer.decode(raw, skip_special_tokens=False),
                'raw_ids': raw, 'complete': bool(raw and raw[-1] == 50),
                'finish_reason': 'complete_batch' if raw and raw[-1] == 50 else 'length',
                'input_tokens': len(tokens), 'max_new_tokens': 512, 'seconds': monotonic() - started,
                'input_sha256': sha256(prompt.encode()).hexdigest()}
    return runtime, generate


def reference_provider(receipt, gpu):
    runtime = reference_model.ReferenceRuntime(receipt, f'cuda:{gpu}')
    def generate(prompt):
        try:
            result = runtime.generate_record(prompt, max_tokens=2048)
        except reference_model.ReferenceContextOverflow as error:
            return {'provider_kind': 'reference', 'raw_text': '', 'raw_ids': [], 'complete': False,
                    'error': 'context_overflow', 'input_tokens': error.input_tokens,
                    'max_new_tokens': error.max_new_tokens, 'context_tokens': error.context_tokens,
                    'input_sha256': sha256(prompt.encode()).hexdigest()}
        return {**result, 'provider_kind': 'reference', 'raw_ids': result['raw_token_ids'],
                'complete': result['finish_reason'] == 'complete_batch'}
    return runtime, generate


def train_worker(rpc, cases, path, generate, name, binding):
    path = history.contained(path, ROOT)
    saved = recover_jsonl(path)
    if len(saved) > len(cases):
        raise ValueError('Saved TRAIN observations exceed the frozen plan')
    with path.open('ab') as stream:
        for index, envelope in enumerate(cases):
            item = envelope['case']
            previous = saved[index] if index < len(saved) else None
            if previous and (previous['case_id'] != item['case_id'] or previous['case_sha256'] != item['case_sha256']):
                raise ValueError('Resumed TRAIN case/prompt identity changed')
            generation = previous['generation'] if previous else generate(item['prompt'])
            if generation.get('diagnostic_worker_sha256') != binding:
                raise ValueError('Saved generation belongs to a different checkpoint or diagnostic worker')
            observation = rpc.request({'action': 'score', 'case': item, 'generation': generation}, 'score')['observation']
            if previous:
                if canonical(previous) != canonical(observation):
                    raise ValueError('Saved TRAIN result no longer reproduces')
            else:
                append_observation(stream, observation)
                if index == 0 or (index + 1) % 25 == 0:
                    print(f'[{name}] TRAIN {index + 1}/{len(cases)}', flush=True)


def dev_worker(manifest, path, gpu, generate, binding):
    path = history.contained(path, ROOT)
    assets = manifest['source']['assets']
    command = ['node', str(ROOT / 'diagnose-dev.mjs'), str(ROOT / assets['registry']['path']),
               str(ROOT / assets['history_dev']['path']), str(path), str(gpu), str(SHARDS), str(RESERVE), binding]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding='utf-8', bufsize=1) as process:
        score = None
        try:
            count = 0
            for line in process.stdout:
                message = json.loads(line)
                if message['kind'] == 'result' and score is None:
                    score = message['score']
                elif message['kind'] == 'generate' and score is None:
                    process.stdin.write(canonical(generate(message['prompt'])) + '\n')
                    process.stdin.flush(); count += 1
                    if count == 1 or count % 25 == 0:
                        print(f'[reference GPU {gpu}] DEV {count} new cases', flush=True)
                else:
                    raise ValueError('Unexpected DEV bridge message')
            process.stdin.close()
            if process.wait() or score is None:
                raise RuntimeError('Reference DEV did not complete; saved prefix remains resumable')
        except BaseException:
            process.kill(); process.wait(); raise


def worker(args):
    import run as runner
    runner.verify_worker_lease(ROOT / 'runs', args.lease_fd)
    if args.parent_fd is None:
        raise ValueError('Workers require a coordinator liveness pipe')
    def parent_watch():
        os.read(args.parent_fd, 1)
        os._exit(72)
    Thread(target=parent_watch, daemon=True).start()
    output = history.contained(ROOT / 'runs' / args.run_id)
    manifest = read(output / 'experiment.json')
    if history.digest(manifest) != read(output / 'manifest-binding.json')['sha256']:
        raise ValueError('Frozen diagnostic manifest changed')
    if args.worker not in (*manifest['students'], 'reference') or args.phase not in ('qualification', 'measure'):
        raise ValueError('Worker is not in the frozen plan')
    if args.phase == 'qualification' and args.worker != 'reference':
        raise ValueError('Only reference transport admission has a qualification phase')
    cases = [row for row in rows(verify(manifest['train_panel']['file']))
             if shard_for(row['case']['case_id']) == args.gpu]
    if args.phase == 'qualification':
        selected = set(manifest['train_panel']['qualification_ids'])
        cases = [row for row in cases if row['case']['case_id'] in selected]
    target = directory(f'runs/{args.run_id}/{args.worker}/{args.phase}')
    binding = worker_binding(manifest, args.worker, args.phase, args.gpu)
    history.immutable(target / f'worker-{args.gpu}.json', {'binding': binding,
        'model': args.worker, 'phase': args.phase, 'gpu': args.gpu,
        'cases': [row['case']['case_id'] for row in cases]})
    if args.worker == 'reference':
        runtime, generate = reference_provider(manifest['reference'], args.gpu)
    else:
        runtime, generate = student_provider(verify(manifest['students'][args.worker]['checkpoint']),
                                            manifest['source']['assets'], args.gpu)
    provider_generate = generate
    def generate(prompt):
        return {**provider_generate(prompt), 'diagnostic_worker_sha256': binding}
    ready = target / f'ready-{args.gpu}-{args.attempt}.json'
    publish_json(ready, {'pid': os.getpid(), 'gpu': args.gpu})
    registry = ROOT / manifest['source']['assets']['registry']['path']
    with NodeRPC('diagnose-data.mjs', registry) as rpc:
        train_worker(rpc, cases, target / f'train-{args.gpu}.jsonl', generate, f'{args.worker} GPU {args.gpu}', binding)
    if args.worker == 'reference' and args.phase == 'measure':
        dev_worker(manifest, target / f'dev-{args.gpu}.jsonl', args.gpu, generate, binding)


def worker_binding(manifest, name, phase, gpu):
    return history.digest({'manifest': history.digest(manifest), 'model': name, 'phase': phase, 'gpu': gpu,
        'weights': manifest['reference']['fingerprint'] if name == 'reference' else manifest['students'][name]['checkpoint']['sha256']})


def launch(output, name, phase, lease):
    target = directory(str(output.relative_to(ROOT) / name / phase))
    attempt = str(os.getpid()) + '-' + str(int(monotonic() * 1000))
    read_fd, write_fd = os.pipe()
    workers = []
    try:
        for gpu in range(SHARDS):
            command = [sys.executable, '-u', str(ROOT / 'diagnose.py'), '--run-id', output.name,
                       '--worker', name, '--phase', phase, '--gpu', str(gpu), '--lease-fd', str(lease),
                       '--parent-fd', str(read_fd), '--attempt', attempt]
            process = subprocess.Popen(command, pass_fds=(lease, read_fd))
            workers.append(process)
            ready = target / f'ready-{gpu}-{attempt}.json'
            while not ready.exists():
                failures = [(index, child.poll()) for index, child in enumerate(workers)
                            if child.poll() is not None and child.returncode != 0]
                if failures or process.poll() is not None:
                    raise RuntimeError(f'Worker failed during model loading: {failures}; saved outputs remain')
                sleep(.25)
        while any(child.poll() is None for child in workers):
            failures = [(index, child.poll()) for index, child in enumerate(workers)
                        if child.poll() is not None and child.returncode != 0]
            if failures:
                raise RuntimeError(f'Diagnostic worker failed: {failures}; repeat the same command to resume')
            sleep(.5)
        if any(child.returncode for child in workers):
            raise RuntimeError('Diagnostic worker failed; no complete report was published')
    finally:
        os.close(write_fd); os.close(read_fd)
        for child in workers:
            if child.poll() is None: child.terminate()
        for child in workers:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


def merged_rows(paths, ordered_ids):
    records = {}
    for path in paths:
        for row in rows(path):
            if row['case_id'] in records:
                raise ValueError('Duplicate result across GPU shards')
            records[row['case_id']] = row
    if set(records) != set(ordered_ids) or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError('GPU shards are missing or adding planned cases')
    return [records[key] for key in ordered_ids]


def metrics(observations):
    eligible = [x for x in observations if x['semantic_eligible']]
    supported = [x for x in eligible if x['cohort'] == 'supported']
    outside = [x for x in eligible if x['cohort'] == 'outside_scope']
    def rate(values, field):
        passed = sum(x.get(field) is True and (field != 'success' or x['critical_status'] == 'none_observed') for x in values)
        return {'passed': passed, 'planned': len(values), 'accuracy': passed / len(values) if values else None}
    return {'supported': rate(supported, 'success'), 'outside_scope': rate(outside, 'success'),
            'native_exact': rate(observations, 'native_exact_match'),
            'canonical_call_exact': rate(observations, 'canonical_call_match'),
            'transport_valid': rate(observations, 'transport_valid'),
            'critical_detected': sum(x['critical_status'] == 'detected' for x in eligible),
            'unresolved': sum(x['critical_status'] == 'review_required' for x in eligible),
            'semantic_ineligible': len(observations) - len(eligible)}


def collect_train(output, manifest, name, phase):
    cases = list(rows(verify(manifest['train_panel']['file'])))
    if phase == 'qualification':
        selected = set(manifest['train_panel']['qualification_ids'])
        cases = [row for row in cases if row['case']['case_id'] in selected]
    target = output / name / phase
    observed = merged_rows([target / f'train-{gpu}.jsonl' for gpu in range(SHARDS)],
                           [row['case']['case_id'] for row in cases])
    for row, envelope in zip(observed, cases):
        if row['case_sha256'] != envelope['case']['case_sha256']:
            raise ValueError('TRAIN result does not belong to the frozen case')
        if row['generation'].get('diagnostic_worker_sha256') != worker_binding(
                manifest, name, phase, shard_for(row['case_id'])):
            raise ValueError('TRAIN observation belongs to a different checkpoint or worker')
    result = {'all_fixed_train_panel': metrics(observed),
              'label_independently_rechecked_here': False,
              'observations': [record(target / f'train-{gpu}.jsonl') for gpu in range(SHARDS)]}
    if name != 'reference':
        seen = [row for row, envelope in zip(observed, cases) if envelope['exposure'][name] > 0]
        unseen = [row for row, envelope in zip(observed, cases) if envelope['exposure'][name] == 0]
        result.update(own_seen_train_fit=metrics(seen), unseen_other_history=metrics(unseen))
    result['by_group'] = {group: metrics([row for row in observed if row['group'] == group])
                          for group in sorted({row['group'] for row in observed})}
    functions = sorted({call['name'] for envelope in cases for call in envelope['case']['gold_calls']})
    result['by_function_overlapping_slices'] = {name: metrics([row for row, envelope in zip(observed, cases)
        if any(call['name'] == name for call in envelope['case']['gold_calls'])]) for name in functions}
    return result


def qualify(output, manifest):
    result = collect_train(output, manifest, 'reference', 'qualification')
    score = result['all_fixed_train_panel']
    admitted = score['transport_valid']['planned'] == PILOT_CASES and score['transport_valid']['passed'] >= PILOT_TRANSPORT_MINIMUM and score['supported']['passed'] > 0
    return {'identity_sha256': manifest['identity_sha256'], 'passed': admitted,
            'criterion': '32 frozen TRAIN-only cases including the longest native prefix; at least30 valid native batches and one complete supported answer',
            'claim': 'transport admission only; measured semantic competence is not established by this gate', **result}


def collect_dev(output, manifest):
    assets = manifest['source']['assets']
    registry, bundle = (ROOT / assets[key]['path'] for key in ('registry', 'history_dev'))
    base_command = ['node', str(ROOT / 'diagnose-dev.mjs'), str(registry), str(bundle)]
    plan = json.loads(subprocess.check_output(base_command + ['--plan'], text=True))
    target = output / 'reference/measure'
    observed = merged_rows([target / f'dev-{gpu}.jsonl' for gpu in range(SHARDS)],
                           [row['case_id'] for row in plan['rows']])
    for row, planned in zip(observed, plan['rows']):
        if row['generation'].get('diagnostic_worker_sha256') != worker_binding(manifest, 'reference', 'measure', planned['shard']):
            raise ValueError('DEV observation belongs to a different reference or worker')
    destination = history.contained(target / 'dev.jsonl', ROOT)
    expected = ''.join(canonical(row) + '\n' for row in observed)
    if destination.exists():
        if destination.read_text() != expected: raise ValueError('Merged DEV changed')
    else:
        with TemporaryDirectory(dir=target) as temporary:
            staged = Path(temporary) / 'dev.jsonl'
            with staged.open('x') as stream:
                stream.write(expected); stream.flush(); os.fsync(stream.fileno())
            os.rename(staged, destination)
    score = json.loads(subprocess.check_output(base_command + ['--summarize', str(destination)], text=True))['score']
    failures = history.infrastructure_failures(observed)
    score.update(infrastructure_failures=len(failures), infrastructure_failure_case_ids=failures,
                 quality_valid=not failures)
    history.validate_score(score, destination, manifest['source']['config'])
    return {'score': score, 'observations': record(destination)}, observed


def paired_cells(student, reference):
    left, right = ({row['case_id']: row for row in values} for values in (student, reference))
    if set(left) != set(right):
        raise ValueError('Paired observations do not contain identical case IDs')
    panels = {}
    for panel in ('fixed_prefix', 'closed_loop', 'challenge'):
        counts = Counter({'both_pass': 0, 'reference_only_pass': 0, 'student_only_pass': 0,
                          'neither_pass': 0, 'unresolved_any': 0})
        for case_id, first in left.items():
            second = right[case_id]
            if first['panel'] != second['panel'] or first['cohort'] != second['cohort']:
                raise ValueError('Paired DEV metadata differs')
            if first['panel'] != panel or first['cohort'] != 'supported': continue
            if 'review_required' in (first['critical_status'], second['critical_status']):
                counts['unresolved_any'] += 1; continue
            a = first.get('success') is True and first['critical_status'] == 'none_observed'
            b = second.get('success') is True and second['critical_status'] == 'none_observed'
            counts['both_pass' if a and b else 'student_only_pass' if a else 'reference_only_pass' if b else 'neither_pass'] += 1
        panels[panel] = dict(counts)
    return panels


def check_inputs():
    import run as runner
    for filename in ('experiments.json', 'experiments-history.json'):
        config = read(ROOT / filename)
        needed = ('base', 'tokenizer', 'registry') if filename == 'experiments.json' else ('history_dev', 'history_replay')
        for key in needed: runner.checked_asset(config['assets'][key])
    context, _ = validate_contexts()
    path = ROOT / 'assets/diagnostic-train-contexts.jsonl'
    print(f'Diagnostic inputs present: context sidecar {path.stat().st_size} bytes; reference {reference_model.MODEL_ID}@{reference_model.REVISION}', flush=True)
    return context


def validate_manifest(manifest, identity, source, output):
    if (manifest.get('identity_sha256') != history.digest(identity)
            or manifest.get('students') != source['students']
            or manifest.get('source') != {'assets': source['manifest']['assets'], 'config': source['manifest']['config']}
            or manifest.get('train_panel') != read(output / 'train-panel.json')
            or manifest.get('no_training_performed') is not True
            or manifest.get('sealed_test_accessed') is not False):
        raise ValueError('Diagnostic manifest changed or belongs to another completed experiment')
    validate_panel(manifest['train_panel'], source)
    reference_model.validate_receipt(manifest['reference'])
    history.immutable(output / 'manifest-binding.json', {'sha256': history.digest(manifest)})


def compare(args):
    import run as runner
    if not SAFE_ID.fullmatch(args.run_id) or len(args.run_id) > 70:
        raise ValueError('Invalid diagnostic run ID')
    runs = directory('runs')
    check_inputs()
    with execution_lease(runs) as lease:
        source = source_context(runs, args.source_run_id)
        output = directory(f'runs/{args.run_id}')
        hardware = runner.four_gpu_identity()
        if version('transformers') != '4.57.3' or version('huggingface-hub') != '0.36.2':
            raise RuntimeError('Use the unchanged pinned requirements.txt for this diagnostic')
        code_paths = BASE_CODE + NEW_CODE + [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'evaluation').glob('*.mjs'))]
        identity = {'source_evidence': source['evidence'], 'code': code_record(ROOT, sorted(set(code_paths))),
                    'hardware': hardware, 'dependencies': {name: version(name) for name in ('transformers', 'huggingface-hub')},
                    'contexts': [record(ROOT / 'assets' / name) for name in ('diagnostic-train-contexts.jsonl', 'diagnostic-train-contexts.json')],
                    'policy': {'seed': SEED, 'gpus': SHARDS, 'student_generation_cap': 512,
                               'reference_generation_cap': 2048, 'qualification_cases': PILOT_CASES,
                               'transport_minimum': PILOT_TRANSPORT_MINIMUM,
                               'model': reference_model.MODEL_ID, 'revision': reference_model.REVISION}}
        history.immutable(output / 'identity.json', identity)
        if (output / 'experiment.json').exists():
            manifest = read(output / 'experiment.json')
            validate_manifest(manifest, identity, source, output)
        else:
            require_space(output, OUTPUT_ALLOWANCE)
            registry = ROOT / source['manifest']['assets']['registry']['path']
            panel_path = output / 'train-panel.json'
            panel = read(panel_path) if panel_path.exists() else prepare_panel(source, output, registry)
            validate_panel(panel, source)
            print('Preparing one pinned Qwen3-8B snapshot inside .runtime; four GPU workers share these files.', flush=True)
            reference = reference_model.prepare_reference(reserve_bytes=RESERVE + OUTPUT_ALLOWANCE)
            require_space(output, OUTPUT_ALLOWANCE)
            manifest = {'schema': 1, 'identity_sha256': history.digest(identity),
                        'source': {'assets': source['manifest']['assets'], 'config': source['manifest']['config']},
                        'students': source['students'], 'train_panel': panel, 'reference': reference,
                        'no_training_performed': True, 'sealed_test_accessed': False,
                        'comparison_type': 'qualified reference recipe, not an architecture-only causal comparison or model ceiling'}
            publish_json(output / 'experiment.json', manifest)
            history.immutable(output / 'manifest-binding.json', {'sha256': history.digest(manifest)})
        final = output / 'result.json'
        if final.exists():
            result = read(final)
            if result.get('identity_sha256') != manifest['identity_sha256']:
                raise ValueError('Final diagnostic identity changed')
            for entry in result['artifacts']: verify(entry)
        stages = directory(f'runs/{args.run_id}/stages')
        qualification = stages / 'reference-transport.json'
        if qualification.exists():
            admitted = read(qualification)
            for entry in admitted['observations']: verify(entry)
            if admitted != qualify(output, manifest):
                raise ValueError('Saved reference transport decision differs from its observations')
        else:
            print('Reference transport admission: 32 TRAIN-only cases on four GPUs.', flush=True)
            launch(output, 'reference', 'qualification', lease)
            admitted = qualify(output, manifest)
            publish_json(qualification, admitted)
        if not admitted['passed']:
            raise RuntimeError(f'Reference transport admission failed. Inspect {qualification}; DEV was not opened to this reference. Do not infer a small-model ceiling.')
        measured = {}
        for name in [*manifest['students'], 'reference']:
            completed = stages / f'{name}.json'
            if completed.exists():
                value = read(completed)
                for entry in value['train']['observations']: verify(entry)
                if name == 'reference': verify(value['dev']['observations'])
                train = collect_train(output, manifest, name, 'measure')
                dev = collect_dev(output, manifest)[0] if name == 'reference' else manifest['students'][name]
                expected = {'identity_sha256': manifest['identity_sha256'], 'model': name, 'train': train,
                            'dev': dev, 'dev_reused_from_authenticated_source': name != 'reference'}
                if value != expected:
                    raise ValueError('Saved model summary differs from its frozen inputs and observations')
            else:
                print(f'Measuring {name}: complete frozen TRAIN union on four GPUs.', flush=True)
                launch(output, name, 'measure', lease)
                train = collect_train(output, manifest, name, 'measure')
                dev = collect_dev(output, manifest)[0] if name == 'reference' else manifest['students'][name]
                value = {'identity_sha256': manifest['identity_sha256'], 'model': name,
                         'train': train, 'dev': dev, 'dev_reused_from_authenticated_source': name != 'reference'}
                publish_json(completed, value)
            measured[name] = value
        reference_rows = list(rows(verify(measured['reference']['dev']['observations'])))
        paired = {name: paired_cells(list(rows(verify(value['dev']['observations']))), reference_rows)
                  for name, value in measured.items() if name != 'reference'}
        artifacts = [record(stages / f'{name}.json') for name in measured] + [record(qualification)]
        artifacts += [record(output / name) for name in ('experiment.json', 'manifest-binding.json', 'train-panel.json', 'train-panel.jsonl')]
        for value in measured.values():
            artifacts += value['train']['observations'] + [value['dev']['observations']]
        for entry in identity['source_evidence']: verify(entry)
        result = {'status': 'COMPLETE_DIAGNOSTIC', 'identity_sha256': manifest['identity_sha256'],
                  'reference_transport': admitted, 'models': measured, 'paired_supported_dev': paired,
                  'artifacts': artifacts, 'release_ready': False,
                  'interpretation': 'TRAIN fit measures consistency with frozen training labels, not independently established scientific truth. Own-history DEV uses each model actual outputs. Unresolved and critical cases cannot count as passes. Reference differences do not establish a model capacity ceiling.'}
        if final.exists():
            if read(final) != result: raise ValueError('Final report differs from recomputed diagnostic summaries')
        else:
            publish_json(final, result)
        print_summary(result, final)
        return result


def print_summary(result, path):
    print(f"Diagnostic: {result['status']}", flush=True)
    for name, value in result['models'].items():
        train = value['train'].get('own_seen_train_fit', value['train']['all_fixed_train_panel'])
        support = train['supported']; dev = value['dev']['score']['summary']
        print(f"{name}: TRAIN supported {support['passed']}/{support['planned']}; "
              f"native {train['native_exact']['passed']}/{train['native_exact']['planned']}; "
              f"DEV supported {dev['supported']['passed']}/{dev['supported']['planned']}; "
              f"dialogues {dev['whole_dialogues']['passed']}/{dev['whole_dialogues']['planned']}; "
              f"DEV critical {dev['critical_detected']}; unresolved {dev['critical_review_required']}", flush=True)
    print(f'Report: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(4,), default=4)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--source-run-id', default='actual-history-v3-4gpu')
    parser.add_argument('--check', action='store_true', help='Check bundled inputs without GPU work or model downloads')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--phase', help=argparse.SUPPRESS)
    parser.add_argument('--gpu', type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--parent-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.check:
        check_inputs()
    else:
        compare(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f'Diagnostic error: {error}')
