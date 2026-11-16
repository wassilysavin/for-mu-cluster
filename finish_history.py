"""Finish the existing frozen seed-1729 pair with no elapsed-time cutoff."""

from private_runtime import configure_private_runtime, private_directory

configure_private_runtime()

import argparse
import base64
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import time
from types import SimpleNamespace

import history as h
import history_parity
from ledger import SAFE_ID, code_record, execution_lease, file_record, publish_json

ROOT = Path(__file__).resolve().parent
COMPLETION_NAME = 'completion-v1'
SEED = 1729
POLICY = {'schema': 1, 'time_limit_seconds': None, 'seeds': [SEED],
          'updates_per_arm': h.UPDATES, 'maximum_cumulative_updates': h.MAX_UPDATES,
          'data_recipe_evaluator_and_worker_identity_unchanged': True,
          'authorization': '2026-11-16: finish both current arms and checks regardless of duration',
          'time_accounting': 'conservative wall time, including uncertain interrupted intervals; not GPU compute time'}


def safe(path):
    return h.contained(path, ROOT)


def read(path):
    return json.loads(safe(path).read_text())


def record(path):
    return file_record(safe(path), relative_to=ROOT)


def verify(entry):
    path = safe(ROOT / entry['path'])
    if record(path) != entry:
        raise ValueError(f"Frozen continuation input changed: {entry['path']}")
    return path


def source_directory(runs, source_id):
    if not SAFE_ID.fullmatch(source_id) or len(source_id) > 70:
        raise ValueError('Invalid source run ID')
    return safe(safe(runs) / source_id / 'revisions' / h.REFERENCE_REVISION / 'recovery' / h.CHECKPOINT_IO_RECOVERY)


def validate_account(account):
    if (account.get('maximum_updates') != h.MAX_UPDATES
            or type(account.get('charged_updates')) is not int
            or not 0 <= account['charged_updates'] <= h.MAX_UPDATES
            or not isinstance(account.get('elapsed_seconds'), (int, float))
            or not math.isfinite(account['elapsed_seconds']) or account['elapsed_seconds'] < 0
            or not isinstance(account.get('stages'), list)):
        raise ValueError('Invalid cumulative work account; it cannot be reset')


class CompletionBudget:
    def __init__(self, path, identity, inherited, *, clock=time):
        self.path, self.clock = safe(path), clock
        validate_account(inherited)
        inherited_hash = h.digest(inherited)
        if self.path.exists():
            value = read(self.path)
            validate_account(value)
            if (value.get('identity_sha256') != identity or value.get('maximum_seconds') is not None
                    or value.get('inherited_sha256') != inherited_hash
                    or value['charged_updates'] < inherited['charged_updates']
                    or value['elapsed_seconds'] < inherited['elapsed_seconds']
                    or value['stages'][:len(inherited['stages'])] != inherited['stages']):
                raise ValueError('Completion account identity or inherited work changed')
        else:
            value = {**json.loads(h.canonical(inherited)), 'identity_sha256': identity,
                     'inherited_sha256': inherited_hash, 'maximum_seconds': None,
                     'prior_time_limit_seconds': inherited['maximum_seconds']}
        if value.get('inflight'):
            phase = value['inflight']
            if not math.isfinite(phase['started_at']):
                raise ValueError('Invalid interrupted phase time')
            seconds = max(0.0, self.clock() - phase['started_at'])
            value['elapsed_seconds'] += seconds
            value['stages'].append({'name': phase['name'], 'seconds': seconds, 'interrupted': True,
                                    'conservative_recovery_charge': True})
            value['inflight'] = None
        self.account = {'path': self.path, 'record': value}
        h.replace_json(self.path, value)

    @property
    def remaining_seconds(self):
        return math.inf

    @property
    def deadline(self):
        return math.inf

    def check(self):
        validate_account(self.account['record'])

    def claim_updates(self, updates):
        self.check()
        value = dict(self.account['record'])
        if type(updates) is not int or updates < 0 or value['charged_updates'] + updates > h.MAX_UPDATES:
            raise h.BudgetLimit('The original optimizer update ceiling is exhausted')
        value['charged_updates'] += updates
        h.replace_json(self.path, value)
        self.account['record'] = value

    @contextmanager
    def phase(self, name):
        self.check()
        value = self.account['record']
        if value.get('inflight'):
            raise RuntimeError('Completion phases must remain sequential')
        value['inflight'] = {'name': name, 'started_at': self.clock()}
        h.replace_json(self.path, value)
        try:
            yield self.deadline
        finally:
            value = self.account['record']
            elapsed = max(0.0, self.clock() - value['inflight']['started_at'])
            value['elapsed_seconds'] += elapsed
            value['stages'].append({'name': name, 'seconds': elapsed,
                                    'interrupted': sys.exc_info()[0] is not None})
            value['inflight'] = None
            h.replace_json(self.path, value)


class CompletionStages:
    def __init__(self, source, output, source_binding, completion_binding, budget, *, verify=lambda: None):
        self.source, self.directory = safe(source), safe(output)
        self.source_binding, self.binding = source_binding, completion_binding
        self.budget, self.verify = budget, verify
        if budget is not None:
            private_directory(ROOT, str((self.directory / 'stages').relative_to(ROOT)))

    def has_receipt(self, name):
        return any(safe(parent / 'stages' / f'{name}.json').is_file()
                   for parent in (self.source, self.directory))

    def run(self, name, action, validate=lambda value: None):
        self.verify()
        original = safe(self.source / 'stages' / f'{name}.json')
        target = safe(self.directory / 'stages' / f'{name}.json')
        for path, binding in ((original, self.source_binding), (target, self.binding)):
            if path.exists():
                saved = read(path)
                if saved.get('identity_sha256') != binding:
                    raise ValueError(f'Stage identity changed: {name}')
                validate(saved['result'])
                return saved['result']
        if self.budget is None:
            raise ValueError(f'Terminal completion is missing its stage receipt: {name}')
        with self.budget.phase(name) as deadline:
            value = action(deadline)
            validate(value)
        publish_json(target, {'identity_sha256': self.binding, 'result': value})
        return value


def prefix_record(path):
    path = safe(path)
    digest = sha256()
    committed = 0
    tail = b''
    with path.open('rb') as stream:
        for line in stream:
            if line.endswith(b'\n'):
                json.loads(line)
                digest.update(line); committed += len(line)
            else:
                tail = line
                if path.name != 'observations.jsonl':
                    continue
                try:
                    json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                digest.update(line); committed += len(line)
    return {'path': str(path.relative_to(ROOT)), 'committed_bytes': committed,
            'committed_sha256': digest.hexdigest(), 'original': record(path),
            'original_tail_base64': base64.b64encode(tail).decode('ascii')}


def verify_prefix(entry):
    path = safe(ROOT / entry['path'])
    remaining, digest = entry['committed_bytes'], sha256()
    with path.open('rb') as stream:
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('An admitted observation/event prefix was truncated')
            digest.update(chunk); remaining -= len(chunk)
    if digest.hexdigest() != entry['committed_sha256']:
        raise ValueError('An admitted observation/event prefix changed')


def validate_checkpoint(source, name, checkpoint, step):
    path = verify(checkpoint)
    metadata = h.checkpoint_header(path)['__metadata__']
    lineage = json.loads(metadata.get('lineage', 'null'))
    expected = source['manifest']['origin']['lineage'] if name == 'baseline' else {
        'identity_sha256': h.digest(source['identities'][name]),
        'initial_checkpoint_sha256': source['manifest']['assets']['initial_checkpoint']['sha256']}
    if int(metadata.get('step', -1)) != step or lineage != expected:
        raise ValueError(f'Wrong checkpoint cursor or lineage for {name}')


def validate_evaluation(source, name, value, *, known=False):
    for key in ('checkpoint', 'observations', 'timings'):
        verify(value[key])
    step = h.INITIAL_STEP if name == 'baseline' else h.INITIAL_STEP + (h.MIDPOINT if known else h.UPDATES)
    validate_checkpoint(source, name, value['checkpoint'], step)
    h.validate_score(value['score'], ROOT / value['observations']['path'], source['manifest']['config'], known=known)
    if sum(row['seconds'] for row in h.timing_records(ROOT / value['timings']['path'])) != value['seconds']:
        raise ValueError('Evaluation timing receipt changed')


def validate_code_identity(current, original):
    if current.get('files') != original.get('files'):
        raise ValueError('Frozen model/training/evaluation code changed')
    for entry in original['files']:
        verify(entry)
    return original


def validate_schedules(schedules, saved_totals):
    totals = {}
    for seed in h.SEEDS:
        for arm in ('control', 'treatment'):
            name = f'{arm}-{seed}'
            rows = schedules[name].rows
            totals[name] = {'prompt': sum(row[2] - row[4] for row in rows),
                            'target': sum(row[4] for row in rows)}
        for field in ('prompt', 'target'):
            left, right = (totals[f'{arm}-{seed}'][field] for arm in ('control', 'treatment'))
            if abs(right - left) > .05 * max(left, right):
                raise ValueError(f'Frozen schedules exceed 5% {field}-token difference')
        left, right = (schedules[f'{arm}-{seed}'].rows for arm in ('control', 'treatment'))
        if sum(a[0] == b[0] for a, b in zip(left, right)) != h.UPDATES * 2:
            raise ValueError('Frozen schedules must retain exactly 50% common replay')
    if any(schedules[f'{arm}-1729'].sha256 == schedules[f'{arm}-2718'].sha256
           for arm in ('control', 'treatment')):
        raise ValueError('Frozen seeds must retain distinct schedules')
    if saved_totals != totals:
        raise ValueError('Frozen token-budget receipt differs from its schedules')


def load_source(runs, source_id):
    import run as runner
    directory = source_directory(runs, source_id)
    manifest = read(directory / 'experiment.json')
    binding = h.digest(manifest)
    config, original, declared = h.load(runner)
    if manifest['config'] != config or manifest['original'] != original:
        raise ValueError('Source protocol changed')
    for key, (_, entry) in declared.items():
        if manifest['assets'].get(key) != entry:
            raise ValueError(f'Source asset binding changed: {key}')
    static = [record(directory / 'experiment.json')]
    static += list(manifest['assets'].values()) + manifest['code']['files']
    static += h.revision_evidence_records(manifest['revision'])
    for value in manifest['origin'].values():
        if isinstance(value, dict) and {'path', 'bytes', 'sha256'} <= value.keys():
            static.append(value)
    for entry in static:
        verify(entry)
    revision = manifest['revision']
    if (revision.get('id') != h.REFERENCE_REVISION
            or revision.get('io_recovery', {}).get('id') != h.CHECKPOINT_IO_RECOVERY):
        raise ValueError('Expected the existing checkpoint-io-v1 recovery')
    budget_path = safe(ROOT / revision['shared_budget']['path'])
    expected_budget = safe(runs / source_id / 'budget.json')
    account = read(budget_path)
    validate_account(account)
    if (budget_path != expected_budget or account.get('identity_sha256') != revision['parent_identity_sha256']
            or revision['shared_budget']['identity_sha256'] != account['identity_sha256']
            or account.get('maximum_seconds') != revision['shared_budget']['maximum_seconds']
            or not 0 < account['maximum_seconds'] <= h.MAX_SECONDS):
        raise ValueError('Original shared budget binding changed')
    h.require_budget_floor(account, revision['budget_at_revision'])
    static.append(record(budget_path))
    stages = {}
    for path in sorted(safe(directory / 'stages').glob('*.json')):
        value = read(path)
        if value.get('identity_sha256') != binding:
            raise ValueError(f'Source stage has a different identity: {path.name}')
        stages[path.stem] = value['result']
        static.append(record(path))
    if not {'official-parity', 'prepare', 'pilot', 'baseline'} <= stages.keys():
        raise ValueError('Completion requires the existing numerical PASS, prepared TRAIN, pilot and full baseline')
    assets = {key: (verify(entry), entry) for key, entry in manifest['assets'].items()}
    parity = read(directory / 'parity/parity.json')
    parity_binding = {key: file_record(assets[asset][0]) for key, asset in (
        ('checkpoint', 'initial_checkpoint'), ('base', 'base'), ('tokenizer', 'tokenizer'),
        ('probes', 'history_replay'), ('native_template', 'history_template'))}
    parity_binding.update(implementation=file_record(ROOT / 'history_parity.py'),
        runtime=file_record(ROOT / 'model.py'), reference_version=history_parity.REFERENCE_VERSION,
        reference_policy=history_parity.reference_policy_record(h.REFERENCE_POLICY), tolerances=history_parity.TOLERANCES)
    if parity != stages['official-parity'] or parity.get('passed') is not True or parity['binding'] != parity_binding:
        raise ValueError('The preserved numerical parity receipt no longer authenticates this runtime')
    static.append(record(directory / 'parity/parity.json'))
    prepared = stages['prepare']
    fingerprint = prepared['collection_receipt']['fingerprint']
    for key, asset in (('checkpoint_sha256', 'initial_checkpoint'), ('registry_sha256', 'registry'),
                       ('tokenizer_sha256', 'tokenizer'), ('specs_sha256', 'history_train_specs')):
        if fingerprint.get(key) != manifest['assets'][asset]['sha256']:
            raise ValueError('TRAIN collection provenance changed')
    train, schedules = h.validate_prepared(prepared, directory, runner)
    validate_schedules(schedules, read(directory / 'token-budget.json'))
    assets['train'] = train
    identities = read(directory / 'identities.json')
    computed, experiments = h.make_identities(runner, original, config, assets, schedules,
                                              manifest['runtime'], manifest['origin'], prepared)
    for name, identity in computed.items():
        identity['code'] = validate_code_identity(identity['code'], identities[name]['code'])
    if computed != identities:
        raise ValueError('Frozen worker identity, recipe or TRAIN membership changed')
    static += [record(directory / 'identities.json'), record(directory / 'token-budget.json'), train[1]]
    source = {'directory': directory, 'manifest': manifest, 'prepared': prepared, 'train': train[0],
              'assets': assets, 'schedules': schedules, 'identities': identities, 'experiments': experiments,
              'stages': stages, 'budget': account, 'source_id': source_id, 'runs': runs, 'static': static,
              'worker_run_id': f'{source_id}-{h.REFERENCE_REVISION}-{h.CHECKPOINT_IO_RECOVERY}'}
    if manifest['assets']['initial_checkpoint'] != manifest['origin']['checkpoint']:
        raise ValueError('The original 7205 checkpoint changed')
    validate_checkpoint(source, 'baseline', manifest['origin']['checkpoint'], h.INITIAL_STEP)
    pilot = stages['pilot']
    validate_checkpoint(source, 'pilot', pilot['checkpoint'], h.INITIAL_STEP + 2)
    measured = runner.require_pilot(runs, source['worker_run_id'] + '-pilot', identities['control-1729'])
    if pilot['measurements'] != measured:
        raise ValueError('Four-GPU pilot measurements changed')
    validate_evaluation(source, 'baseline', stages['baseline'])
    for arm in ('control', 'treatment'):
        name = f'{arm}-{SEED}'
        for suffix, milestone in (('midpoint', h.MIDPOINT), ('trained', h.UPDATES)):
            if f'{name}-{suffix}' in stages:
                value = stages[f'{name}-{suffix}']
                validate_checkpoint(source, name, value['checkpoint'], h.INITIAL_STEP + milestone)
        for suffix, known in (('regression', True), ('final', False)):
            if f'{name}-{suffix}' in stages:
                validate_evaluation(source, name, stages[f'{name}-{suffix}'], known=known)
        worker = safe(runs / f"{source['worker_run_id']}-{name}")
        latest = safe(worker / 'checkpoint.safetensors')
        if latest.exists():
            step = int(h.checkpoint_header(latest)['__metadata__']['step'])
            if not h.INITIAL_STEP <= step <= h.INITIAL_STEP + h.UPDATES:
                raise ValueError('Source worker cursor is outside the frozen schedule')
            validate_checkpoint(source, name, record(latest), step)
        for known in (True, False):
            key = name + ('-regression' if known else '')
            path = safe(directory / 'evaluations' / key / 'input.json')
            if path.exists():
                value = read(path)
                step = h.INITIAL_STEP + (h.MIDPOINT if known else h.UPDATES)
                validate_checkpoint(source, name, value['checkpoint'], step)
                if value != {'run': identities[name], 'checkpoint': value['checkpoint'], 'known': known}:
                    raise ValueError('Partial DEV belongs to another model or checkpoint')
                static.append(record(path))
    final = safe(directory / 'result.json')
    if final.exists():
        result = read(final)
        if result.get('identity_sha256') != binding or result.get('baseline') != stages['baseline']:
            raise ValueError('Original terminal report changed or is misbound')
        if result.get('status') not in {'budget_stopped', 'not_promoted', 'inconclusive', 'diagnostic_gain',
                'exploratory_gain_second_seed_not_run', 'confirmed_diagnostic_gain'}:
            raise ValueError('Unexpected source terminal status')
        static.append(record(final))
    source['result_was_present'] = final.exists()
    return source


def snapshot_source(source):
    static = {entry['path']: entry for entry in source['static']}
    prefixes, checkpoints = [], []
    for arm in ('control', 'treatment'):
        name = f'{arm}-{SEED}'
        worker = safe(source['runs'] / f"{source['worker_run_id']}-{name}")
        for path in sorted(worker.glob('checkpoint-step-*.safetensors')):
            static[str(safe(path).relative_to(ROOT))] = record(path)
        for filename in ('run.json',):
            path = safe(worker / filename)
            if path.exists(): static[str(path.relative_to(ROOT))] = record(path)
        latest = safe(worker / 'checkpoint.safetensors')
        if latest.exists():
            checkpoints.append({'file': record(latest), 'name': name,
                                'step': int(h.checkpoint_header(latest)['__metadata__']['step'])})
        events = safe(worker / 'events.jsonl')
        if events.exists(): prefixes.append(prefix_record(events))
        for key in (name, name + '-regression'):
            for filename in ('observations.jsonl', 'timings.jsonl'):
                path = safe(source['directory'] / 'evaluations' / key / filename)
                if path.exists(): prefixes.append(prefix_record(path))
    return {'immutable': list(static.values()), 'append_only_prefixes': prefixes,
            'admitted_latest_checkpoints': checkpoints,
            'original_result_present': source['result_was_present']}


def validate_completion(manifest, source, output):
    if (manifest.get('policy') != POLICY or manifest.get('source_manifest') != record(source['directory'] / 'experiment.json')
            or manifest.get('source_directory') != str(source['directory'].relative_to(ROOT))
            or manifest.get('inherited_budget') != source['budget']
            or manifest.get('code', {}).get('files') != code_record(ROOT, ['finish_history.py'])['files']):
        raise ValueError('Completion protocol or source binding changed')
    for entry in manifest['source_snapshot']['immutable']:
        verify(entry)
    for entry in manifest['source_snapshot']['append_only_prefixes']:
        verify_prefix(entry)
    for admitted in manifest['source_snapshot']['admitted_latest_checkpoints']:
        latest = safe(ROOT / admitted['file']['path'])
        step = int(h.checkpoint_header(latest)['__metadata__']['step'])
        if not admitted['step'] <= step <= h.INITIAL_STEP + h.UPDATES:
            raise ValueError('An admitted training cursor was rolled back or exceeded its frozen schedule')
        validate_checkpoint(source, admitted['name'], record(latest), step)
        if step == admitted['step'] and record(latest) != admitted['file']:
            raise ValueError('An admitted checkpoint changed without advancing its cursor')
    if safe(source['directory'] / 'result.json').exists() != manifest['source_snapshot']['original_result_present']:
        raise ValueError('The original coordinator changed after completion was admitted')
    binding = read(output / 'manifest-binding.json')
    if binding != {'sha256': h.digest(manifest)}:
        raise ValueError('Completion manifest changed')


def make_backend(source, budget, stages):
    import run as runner
    args = SimpleNamespace(runs=source['runs'], run_id=source['worker_run_id'])
    manifest = source['manifest']
    backend = h.ClusterBackend(runner, args, manifest['config'], manifest['original'], source['assets'],
        source['directory'], manifest['origin'], manifest['runtime'], budget, stages)
    backend.identities, backend.experiments = source['identities'], source['experiments']
    backend.schedules = source['schedules']
    return backend


def complete_pair(backend, stages, budget, initial):
    pending = sum(backend.remaining_updates(f'{arm}-{SEED}') for arm in ('control', 'treatment'))
    if budget.account['record']['charged_updates'] + pending > h.MAX_UPDATES:
        raise h.BudgetLimit('The remaining pair cannot fit the original optimizer update ceiling')
    if not all(stages.has_receipt(f'{arm}-{SEED}-final') for arm in ('control', 'treatment')):
        backend.space(evaluations=2)
    baseline = stages.run('baseline', lambda deadline: backend.evaluate('baseline', initial, deadline),
                          backend.validate_evaluation)
    results = {}
    for arm in ('control', 'treatment'):
        name = f'{arm}-{SEED}'
        midpoint = stages.run(name + '-midpoint', lambda deadline: backend.train(name, h.MIDPOINT, deadline),
                              lambda value: backend.validate_artifact(value['checkpoint']))
        stages.run(name + '-regression', lambda deadline: backend.evaluate(name,
            ROOT / midpoint['checkpoint']['path'], deadline, known=True),
            lambda value: backend.validate_evaluation(value, known=True))
        trained = stages.run(name + '-trained', lambda deadline: backend.train(name, h.UPDATES, deadline),
                             lambda value: backend.validate_artifact(value['checkpoint']))
        results[arm] = stages.run(name + '-final', lambda deadline: backend.evaluate(name,
            ROOT / trained['checkpoint']['path'], deadline), backend.validate_evaluation)
    outcome = h.decision(baseline, results['control'], results['treatment'],
                         (backend.outcomes(results['control']), backend.outcomes(results['treatment'])))
    pair = {'seed': SEED, **results, 'decision': outcome}
    return {'status': 'exploratory_gain_second_seed_not_run' if outcome['passed'] else outcome['status'],
            'baseline': baseline, 'pairs': [pair], 'exploratory': True, 'product_acceptance': False,
            'selection': {'status': 'retain_7205_as_diagnostic_reference', 'checkpoint': baseline['checkpoint'],
                          'rule': 'Complete the requested first pair; no confirmation seed or product acceptance'}}


def completion_artifacts(output):
    paths = [output / name for name in ('experiment.json', 'manifest-binding.json', 'budget.json')]
    paths += sorted((output / 'stages').glob('*.json'))
    return [record(path) for path in paths]


def load_completed_source(runs, source_id, *, require_quality=True):
    source = load_source(runs, source_id)
    output = safe(source['directory'] / COMPLETION_NAME)
    manifest = read(output / 'experiment.json')
    validate_completion(manifest, source, output)
    result = read(output / 'result.json')
    if (result.get('identity_sha256') != h.digest(manifest) or result.get('completion_policy') != POLICY
            or len(result.get('pairs', [])) != 1 or result['pairs'][0].get('seed') != SEED
            or result.get('TEST_opened') is not False or result.get('product_acceptance') is not False
            or result.get('exploratory') is not True or result.get('original_budget') != source['budget']):
        raise ValueError('Completion result identity, target pair or policy changed')
    budget = read(output / 'budget.json')
    validate_account(budget)
    if (budget.get('identity_sha256') != h.digest(manifest) or budget.get('inherited_sha256') != h.digest(source['budget'])
            or budget.get('maximum_seconds') is not None or budget.get('inflight') is not None
            or budget['charged_updates'] < source['budget']['charged_updates']
            or budget['elapsed_seconds'] < source['budget']['elapsed_seconds']
            or budget['stages'][:len(source['budget']['stages'])] != source['budget']['stages']
            or result.get('budget') != budget):
        raise ValueError('Completed work accounting changed')
    if result.get('artifacts') != completion_artifacts(output):
        raise ValueError('Completion artifact inventory changed')
    stages = CompletionStages(source['directory'], output, h.digest(source['manifest']), h.digest(manifest),
                              None, verify=lambda: validate_completion(manifest, source, output))
    def no_action(*unused):
        raise ValueError('A terminal completion is missing its complete stage receipts')
    baseline = stages.run('baseline', no_action, lambda value: validate_evaluation(source, 'baseline', value))
    arms = {}
    for arm in ('control', 'treatment'):
        name = f'{arm}-{SEED}'
        for suffix, milestone in (('midpoint', h.MIDPOINT), ('trained', h.UPDATES)):
            stages.run(name + '-' + suffix, no_action, lambda value: validate_checkpoint(
                source, name, value['checkpoint'], h.INITIAL_STEP + milestone))
        stages.run(name + '-regression', no_action, lambda value: validate_evaluation(source, name, value, known=True))
        arms[arm] = stages.run(name + '-final', no_action, lambda value: validate_evaluation(source, name, value))
    if result['baseline'] != baseline or any(result['pairs'][0][arm] != arms[arm] for arm in arms):
        raise ValueError('Completion report differs from authenticated evaluations')
    observed = [h.read_observations(verify(arms[arm]['observations'])) for arm in ('control', 'treatment')]
    outcomes = [{row['case_id']: row for row in rows} for rows in observed]
    outcome = h.decision(baseline, arms['control'], arms['treatment'], outcomes)
    status = 'exploratory_gain_second_seed_not_run' if outcome['passed'] else outcome['status']
    if result['status'] != status or result['pairs'][0]['decision'] != outcome:
        raise ValueError('Completion comparison no longer reproduces')
    students = {'baseline-7205': baseline, **{f'{arm}-{SEED}': arms[arm] for arm in arms}}
    for name, evaluated in students.items():
        if require_quality and evaluated['score'].get('infrastructure_failures'):
            raise ValueError(f'{name} has source evaluation infrastructure failures')
    evidence = source['static'] + result['artifacts'] + [record(output / 'result.json')]
    for candidate in students.values():
        evidence.extend(candidate[key] for key in ('checkpoint', 'observations', 'timings'))
    return {'directory': source['directory'], 'manifest': source['manifest'], 'result': result,
            'prepared': source['prepared'], 'train': source['train'], 'students': students,
            'evidence': list({entry['path']: entry for entry in evidence}.values())}


def compare(args):
    import run as runner
    runs = safe(args.runs)
    with execution_lease(runs):
        source = load_source(runs, args.source_run_id)
        output = safe(source['directory'] / COMPLETION_NAME)
        if safe(output / 'result.json').exists():
            completed = load_completed_source(runs, args.source_run_id, require_quality=False)
            h.print_summary(completed['result'], output / 'result.json')
            return completed['result']
        hardware = runner.four_gpu_identity()
        if hardware != source['manifest']['runtime']:
            raise ValueError('Use the original Python environment and four allocated GPUs')
        runner.require_run_space(runs, 1024 * 1024)
        output = private_directory(ROOT, str(output.relative_to(ROOT)))
        path = output / 'experiment.json'
        if path.exists():
            manifest = read(path)
        else:
            manifest = {'schema': 1, 'policy': POLICY, 'source_directory': str(source['directory'].relative_to(ROOT)),
                        'source_manifest': record(source['directory'] / 'experiment.json'),
                        'inherited_budget': source['budget'], 'source_snapshot': snapshot_source(source),
                        'code': code_record(ROOT, ['finish_history.py'])}
            publish_json(path, manifest)
        h.immutable(output / 'manifest-binding.json', {'sha256': h.digest(manifest)})
        validate_completion(manifest, source, output)
        budget = CompletionBudget(output / 'budget.json', h.digest(manifest), source['budget'])
        stages = CompletionStages(source['directory'], output, h.digest(source['manifest']), h.digest(manifest),
                                  budget, verify=lambda: validate_completion(manifest, source, output))
        backend = make_backend(source, budget, stages)
        print('Finishing the existing seed-1729 pair; no elapsed-time cutoff. Frozen steps, data and resource checks remain.', flush=True)
        for arm in ('control', 'treatment'):
            print(f'{arm}: {backend.remaining_updates(f"{arm}-{SEED}")} training updates remain before final DEV.', flush=True)
        result = complete_pair(backend, stages, budget, source['assets']['initial_checkpoint'][0])
        validate_completion(manifest, source, output)
        artifacts = completion_artifacts(output)
        result.update(identity_sha256=h.digest(manifest), completion_policy=POLICY,
                      budget=budget.account['record'], original_budget=manifest['inherited_budget'],
                      artifacts=artifacts, TEST_opened=False)
        publish_json(output / 'result.json', result)
        h.print_summary(result, output / 'result.json')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(4,), default=4)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--source-run-id', default='actual-history-v3-4gpu')
    args = parser.parse_args()
    compare(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f'Completion error: {error}')
