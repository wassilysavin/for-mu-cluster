"""Print a paired audit of the saved FunctionGemma DEV evaluations."""

import argparse
from collections import Counter
import sys

from .student_evidence import load_saved, validate_replay
from .student_errors import inspectArm
from eval_tools.diagnose_dev import loadPanel


ARMS = ('baseline-7205', 'control-1729', 'treatment-1729')
PANELS = ('fixed_prefix', 'closed_loop', 'challenge')


def load_report():
    registry, bundle, arms = load_saved()
    if set(arms) != set(ARMS):
        raise ValueError('Expected the baseline and both seed-1729 evaluations')
    context = loadPanel(registry, bundle)
    results = {}
    for arm in ARMS:
        saved_score, observations = arms[arm]
        result = inspectArm(context, observations)
        validate_replay(saved_score, result['score'])
        results[arm] = result
    return results


def paired_supported(results, panel):
    by_arm = {}
    for arm in ARMS:
        cases = [row for row in results[arm]['rows']
                 if row['panel'] == panel and row['cohort'] == 'supported']
        by_arm[arm] = {row['case_id']: row for row in cases}
        expected = results[arm]['score']['panels'][panel]['supported']
        if (len(cases) != expected['planned'] or len(by_arm[arm]) != len(cases)
                or sum(row['success'] is True for row in cases) != expected['passed']):
            raise ValueError(f'{arm}: supported cases disagree with replayed totals')
        if by_arm[arm].keys() != by_arm[ARMS[0]].keys():
            raise ValueError('The arms have different planned supported cases')
    return by_arm


def counts(items, limit=None):
    ranked = sorted(Counter(items).items(), key=lambda pair: (-pair[1], pair[0]))
    return ', '.join(f'{key}={n}' for key, n in ranked[:limit]) or 'none'


def example_line(row):
    def names(batch):
        return ' + '.join(batch) if batch else '(no decoded calls)'

    reference = ''
    if row['failure_category'] in {'omitted_reference', 'wrong_reference'}:
        selected = ','.join(row['selected_targets']) or 'none'
        expected = ','.join(row['first_allowed_targets']) or 'none'
        reference = f' (selected {selected}, first allowed {expected})'
    return (f"  {row['case_id']} [{row['failure_category']}]: "
            f"got {names(row['actual_names'])}; first allowed alternative "
            f"{names(row['expected_names'][0])}{reference}")


def render(results, panel='closed_loop'):
    by_arm = paired_supported(results, panel)
    lines = ['FunctionGemma | saved DEV1250 | CPU replay',
             'arm             supported    dialogues   outside   critical/review   all critical/review    failed: fixed/loop/challenge']
    for arm in ARMS:
        panels = results[arm]['score']['panels']
        loop = panels['closed_loop']
        def fraction(key):
            return f"{loop[key]['passed']}/{loop[key]['planned']}"
        fails = '/'.join(str(sum(not row['success'] for row in results[arm]['rows']
                                 if row['panel'] == panel)) for panel in PANELS)
        total_critical = sum(panels[key]['critical_detected'] for key in PANELS)
        total_review = sum(panels[key]['critical_review_required'] for key in PANELS)
        lines.append(f"{arm:16}{fraction('supported'):13}{fraction('whole_dialogues'):12}"
                     f"{fraction('outside_scope'):10}{loop['critical_detected']}/{loop['critical_review_required']:<13}"
                     f"{total_critical}/{total_review:<19}{fails}")
    if any(results[arm]['score'].get('infrastructure_failures') for arm in ARMS):
        lines.append('Infrastructure failures: ' + ', '.join(
            f"{arm}={results[arm]['score']['infrastructure_failures']}" for arm in ARMS))
    lines.append('Supported, dialogues, outside and first critical/review use closed-loop only.')
    lines.append(f'Errors on {panel} supported cases:')
    for arm, cases in by_arm.items():
        failed = [row for row in cases.values() if not row['success']]
        lines.append(f"  {arm} ({len(failed)}): " + counts(row['failure_category'] for row in failed))
        lines.append('    call differences: ' + counts(signal for row in failed for signal in row['signals']))
    control, treatment = (by_arm[arm] for arm in ARMS[1:])
    gained = sum(not control[key]['success'] and row['success'] for key, row in treatment.items())
    lost = sum(control[key]['success'] and not row['success'] for key, row in treatment.items())
    lines.append(f'Control → treatment, same scenario/turn: {gained} gained, {lost} lost (each arm used its own history).')
    failed = [row for row in treatment.values() if not row['success']]
    lines.append('Treatment call comparisons: ' + counts(row['comparison'] for row in failed))
    lines.append('Treatment execution codes: ' + counts(row['error_code'] for row in failed if row['error_code']))
    lines.append('Treatment argument fields (top 5): ' + counts(
        (field for row in failed for field in set(row['argument_fields'])), limit=5))
    by_group = Counter(row['group'] for row in treatment.values())
    failed_groups = Counter(row['group'] for row in failed)
    top_groups = sorted(failed_groups, key=lambda key: (-failed_groups[key], key))[:3]
    if top_groups:
        lines.append('Treatment groups with most failures (failed/planned): ' + ', '.join(
            f'{key} {failed_groups[key]}/{by_group[key]}' for key in top_groups))
    candidates = [row for row in sorted(failed, key=lambda row: row['case_id'])
                  if row['actual_names'] and row['expected_names']]
    chosen = []
    for category in ('wrong_reference', 'omitted_reference', 'execution_error', 'semantic_unresolved',
                     'unnecessary_clarification'):
        match = next((row for row in candidates if row['failure_category'] == category), None)
        if match:
            chosen.append(match)
        if len(chosen) == 2:
            break
    if chosen:
        lines.append('Treatment examples (operation names only):')
        lines.extend(map(example_line, chosen))
    lines.append('Call differences can overlap. Unresolved cases retain their recorded grades.')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=PANELS, default='closed_loop',
                        help='panel for supported error details (default: closed_loop)')
    args = parser.parse_args()
    try:
        print(render(load_report(), args.panel))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        sys.exit(f'Cannot verify the saved FunctionGemma comparison: {error}')
