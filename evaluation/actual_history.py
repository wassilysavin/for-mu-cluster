"""Resolve and score DEV continuations against each candidate's own displayed history."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256

from .schema import js_json


def hash(value):
    return sha256((value if isinstance(value, str) else js_json(value)).encode()).hexdigest()


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def refs(calls):
    keys = ("target", "_replace", "_resolve", "_supersede")
    return [c["arguments"][k] for c in calls for k in keys if isinstance(c.get("arguments", {}).get(k), str)]


class ActualHistoryHarness:
    def __init__(self, api):
        self.api = api

    refs = staticmethod(refs)

    def visibleHash(self, before):
        history = {k: v for k, v in self.api.sessionHistory(before).items() if k != "session_id"}
        return hash(self.api.canonical(history))

    def resolve(self, row, before):
        policy = row["policy"]
        kind = policy["kind"]
        index = policy.get("target_exchange") or 0
        ex = before["exchanges"][index - 1] if 0 < index <= len(before["exchanges"]) else {}
        turn = next((t for t in before["semantic"]["turns"] if t["id"] == ex.get("semantic_turn_id")), {})
        answers = turn.get("answers", [])
        part = policy.get("part", 1)
        answer = answers[part - 1] if 0 < part <= len(answers) else {}
        branch, targets = "default", []
        if kind == "fresh":
            batches = policy["allowed_batches"] if "allowed_batches" in policy else [policy["calls"]]
        elif kind in ("quote", "repeat_answer"):
            item, noun = (ex, "exchange") if kind == "quote" else (answer, "answer")
            branch = "present" if item else "absent"
            if item:
                targets = [item["id"]]
                batches = [[_call("repeat_" + noun, target=item["id"])]]
            else:
                batches = [[_call("clarify_reference", kind=noun)]]
        elif kind == "retry":
            failed = ex.get("outcome") in ("error", "provider_error")
            branch = "failed" if failed else "answered"
            targets = [ex["id"]] if failed else []
            batches = [
                [*([_call("retry_exchange", target=ex["id"])] if failed else []), *deepcopy(policy["calls"])]
            ]
        elif kind == "replace":
            branch = "present" if answer else "absent"
            fresh = deepcopy(policy["calls"])
            batches = [fresh]
            if answer:
                targets = [answer["id"]]
                original = answer["call"]
                if len(fresh) == 1 and self.api.canonical(original) == self.api.canonical(fresh[0]):
                    branch = "already_correct"
                    batches = [[_call("repeat_answer", target=answer["id"])]]
                else:
                    fresh[0]["arguments"]["_replace"] = answer["id"]
                    if len(fresh) == 1 and original["name"] == fresh[0]["name"]:
                        old, new = original["arguments"], policy["calls"][0]["arguments"]
                        changes = {
                            k: v
                            for k, v in new.items()
                            if k not in old or self.api.canonical(old[k]) != self.api.canonical(v)
                        }
                        remove = [k for k in old if k not in new]
                        batches.append(
                            [
                                _call(
                                    "change_" + original["name"],
                                    target=answer["id"],
                                    changes=changes,
                                    **({"remove": remove} if remove else {}),
                                )
                            ]
                        )
        elif kind in ("pending_cancel", "pending_resolve", "pending_supersede"):
            ids = {p["id"] for p in turn.get("response_parts", []) if p["kind"] == "clarification"}
            pending = next(
                (
                    p
                    for p in before["semantic"]["pending"]
                    if p["id"] in ids and (not policy.get("function") or p["function"] == policy["function"])
                ),
                None,
            )
            branch = "present" if pending else "absent"
            if pending:
                target = pending["id"]
                targets = [target]
                if kind == "pending_cancel":
                    batches = [[_call("cancel_pending", target=target)]]
                elif kind == "pending_resolve":
                    batches = [[_call(pending["function"], **policy["values"], _resolve=target)]]
                else:
                    fresh = deepcopy(policy["calls"][0])
                    fresh["arguments"]["_supersede"] = target
                    batches = [[fresh]]
            elif kind == "pending_cancel":
                batches = [[_call("clarify_reference", kind="pending")]]
            else:
                batches = [deepcopy(policy.get("fallback_calls") or policy["calls"])]
        elif kind == "ambiguous":
            batches = [[_call("clarify_reference", kind=policy["object_kind"])]]
        else:
            raise ValueError(f"Unknown policy {kind}")
        parts = turn.get("response_parts", [])
        displayed = (
            ex.get("outcome") == "committed"
            and answers
            and len(answers) == len(parts)
            and all(p["kind"] == "answer" and p["id"] == a["id"] for p, a in zip(parts, answers))
            and "\n\n".join(p["text"] for p in parts) == ex["assistant"]
        )
        if kind == "quote" and displayed:
            batches.append([_call("repeat_answer", target=a["id"]) for a in answers])
        if (kind == "repeat_answer" or branch == "already_correct") and answer and displayed and len(answers) == 1:
            batches.append([_call("repeat_exchange", target=ex["id"])])
        question = row.get("questions", {}).get(branch, row.get("question"))
        if not question or not batches or not all(batches):
            raise ValueError(f"Incomplete oracle {row.get('case_id')}/{branch}")
        result = dict(
            question=question,
            branch=branch,
            allowed_batches=batches,
            reference_targets=targets,
            reference_target_alternatives=[refs(batch) for batch in batches],
        )
        if "reference_eligible" in row:
            result.update(
                reference_eligible=row["reference_eligible"] and bool(targets),
                authored_reference_eligible=row["reference_eligible"],
            )
        return dict(result, before_visible_history_sha256=self.visibleHash(before))

    def initial(self, registry, row):
        sid = "dev-" + (row.get("dialogue_id") or row["case_id"])
        state = self.api.createSession(registry, dict(dialogueVersion=2, sessionId=sid))
        for entry in row.get("initial_prefix", []):
            if entry.get("policy"):
                resolved = self.resolve(entry, state)
                question = resolved["question"]
                raw = self.api.encodeCalls(resolved["allowed_batches"][0])
            else:
                question = entry["question"]
                raw = entry.get("raw_output")
                if raw is None:
                    raw = self.api.encodeCalls(entry["calls"])
            state = self.api.applyModelOutput(registry, state, question, raw)
            if state["exchanges"][-1]["outcome"] != entry.get("expected_outcome"):
                raise ValueError(f"Prefix outcome mismatch {row['case_id']}")
        return state

    def grade(self, registry, row, before, after, resolved):
        ex = after["exchanges"][-1]
        question = resolved["question"]
        expected = [
            self.api.applyModelOutput(registry, before, question, self.api.encodeCalls(batch))
            for batch in resolved["allowed_batches"]
        ]
        def equal(left, right):
            return self.api.canonical(left) == self.api.canonical(right)

        for state in expected:
            if state["exchanges"][-1]["outcome"] != "committed":
                raise ValueError(f"Oracle execution failed {row['case_id']}/{resolved['branch']}")
            if row["policy"]["kind"] in ("quote", "repeat_answer") or resolved["branch"] == "already_correct":
                if (
                    not equal(state["semantic"]["pending"], before["semantic"]["pending"])
                    or not equal(state["semantic"]["turns"][:-1], before["semantic"]["turns"])
                    or state["exchanges"][-1]["assistant"] != expected[0]["exchanges"][-1]["assistant"]
                ):
                    raise ValueError(f"Incomplete quote alternative {row['case_id']}")
        integrity = (
            len(after["exchanges"]) == len(before["exchanges"]) + 1
            and equal(after["exchanges"][:-1], before["exchanges"])
            and ex["question"] == question
        )
        calls = ex.get("model_calls") or []
        selected, targets = refs(calls), resolved["reference_targets"]
        alternatives = resolved["reference_target_alternatives"]
        correct = selected in alternatives
        committed = ex["outcome"] == "committed"
        passed = (
            integrity
            and committed
            and any(
                ex["assistant"] == s["exchanges"][-1]["assistant"] and equal(after["semantic"], s["semantic"])
                for s in expected
            )
        )

        def clarifies(c):
            return c["name"] == "clarify_reference" or c["arguments"].get("_clarify")

        status, category, failure = (
            ("none_observed", None, None) if passed else ("review_required", None, "semantic_unresolved")
        )
        if not integrity:
            status, category = "detected", "history_integrity"
        elif not committed:
            status, failure = "none_observed", "execution_error"
        elif not passed and targets and selected and not correct:
            status, category, failure = "detected", "wrong_reference", "wrong_reference"
        elif (
            not passed
            and any(map(clarifies, calls))
            and not any(clarifies(c) for b in resolved["allowed_batches"] for c in b)
        ):
            status, failure = "none_observed", "unnecessary_clarification"
        elif not passed and targets and not selected and any(not c["arguments"].get("_clarify") for c in calls):
            status, category, failure = "detected", "omitted_reference", "omitted_reference"
        eligible = row["reference_eligible"]
        return dict(
            case_id=row["case_id"],
            family_id=row["family_id"],
            bootstrap_family_id=row.get("bootstrap_family_id", row["family_id"]),
            panel=row["panel"],
            cohort=row["cohort"],
            dialogue_id=row.get("dialogue_id"),
            turn_index=row["turn_index"],
            standalone=row["cohort"] == "supported" and not row.get("dialogue_id"),
            success=passed,
            critical_status=status,
            critical_category=category,
            failure_category=failure,
            reference_eligible=eligible,
            resolved_reference_available=bool(targets),
            reference_success=(bool(targets) and correct and committed) if eligible else None,
            expected_reference_targets=targets,
            expected_reference_alternatives=alternatives,
            selected_reference_targets=selected,
        )

    @staticmethod
    def summarize(rows, results):
        from .evaluation import metric

        by_id = {r["case_id"]: r for r in results}
        groups = defaultdict(list)
        episodes = defaultdict(list)
        for row in rows:
            cid = row["case_id"]
            groups[row["cohort"]].append(cid)
            if row.get("dialogue_id"):
                episodes[row["dialogue_id"]].append(cid)
            elif row["cohort"] == "supported":
                groups["standalone"].append(cid)
            if row.get("reference_eligible"):
                groups["exact_reference"].append(cid)

        def count(group, key="success"):
            ids = groups[group]
            return metric(len(ids), sum(by_id.get(cid, {}).get(key) is True for cid in ids))

        whole = sum(all(by_id.get(cid, {}).get("success") is True for cid in ids) for ids in episodes.values())
        critical = Counter(
            (r.get("critical_category") or "other") for r in results if r["critical_status"] == "detected"
        )
        hist = sum(critical[k] for k in ("wrong_reference", "omitted_reference", "history_integrity"))
        return dict(
            supported=count("supported"),
            outside_scope=count("outside_scope"),
            whole_dialogues=metric(len(episodes), whole),
            standalone=count("standalone"),
            exact_reference=count("exact_reference", "reference_success"),
            missing=sum(r["case_id"] not in by_id for r in rows),
            critical_detected=sum(critical.values()),
            critical_review_required=sum(r["critical_status"] == "review_required" for r in results),
            history_critical=hist,
            critical_types=dict(critical),
            critical_by_kind=dict(history=hist, other=sum(critical.values()) - hist),
            release_ready=False,
        )


def createActualHistoryHarness(api):
    return ActualHistoryHarness(api)
