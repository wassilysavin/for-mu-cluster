from copy import deepcopy
import unittest

import inspect_student_errors as inspector


def observation(case, passed, panel='closed_loop', group='scope'):
    return dict(case_id=case, success=passed, panel=panel, cohort='supported', group=group,
                failure_category=None if passed else 'omitted_reference',
                signals=[] if passed else ['operation_substitution'], argument_fields=[],
                error_code=None, comparison='passed' if passed else 'different',
                actual_names=['get_units'], expected_names=[['repeat_answer']],
                selected_targets=[], first_allowed_targets=['t2.c1'])


def arm(outcomes):
    rows = [observation(f'c{i}', passed) for i, passed in enumerate(outcomes)]
    rows += [observation('fixed', False, 'fixed_prefix'), observation('challenge', False, 'challenge')]
    panels = {}
    for panel in inspector.PANELS:
        selected = [row for row in rows if row['panel'] == panel]
        panels[panel] = dict(supported=dict(passed=sum(row['success'] for row in selected), planned=len(selected)),
                             whole_dialogues=dict(passed=0, planned=0), outside_scope=dict(passed=0, planned=0),
                             critical_detected=sum(not row['success'] for row in selected), critical_review_required=0)
    return {'score': {'panels': panels}, 'rows': rows}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.results = dict(zip(inspector.ARMS, (arm([False, False, True]), arm([False, False, True]),
                                                 arm([True, True, False]))))

    def test_gains_and_losses_join_on_case_not_position(self):
        self.results['treatment-1729']['rows'].reverse()
        text = inspector.render(self.results)
        self.assertIn('2 gained, 1 lost', text)
        self.assertIn('treatment-1729 (1): omitted_reference=1', text)
        self.assertIn('(selected none, first allowed t2.c1)', text)

    def test_challenge_does_not_count_closed_loop_errors(self):
        text = inspector.render(self.results, 'challenge')
        self.assertIn('control-1729 (1): omitted_reference=1', text)
        self.assertIn('0 gained, 0 lost', text)

    def test_unpaired_and_duplicate_supported_cases_fail(self):
        for invalid_id in ('not-in-plan', 'c1'):
            with self.subTest(invalid_id=invalid_id):
                results = deepcopy(self.results)
                results['control-1729']['rows'][0]['case_id'] = invalid_id
                with self.assertRaises(ValueError):
                    inspector.render(results)


if __name__ == '__main__':
    unittest.main()
