"""Authored dialogue contrasts with verified FunctionGemma training targets."""

from copy import deepcopy

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def turn(question, *calls):
    return {"question": question, "calls": list(calls), "expected_outcome": "committed"}


def failed(core, step):
    invalid = call("get_units", subject="unlisted_coordinate_scale")
    return {
        "question": step["question"],
        "raw_output": core.encodeCalls([invalid]),
        "expected_outcome": "error",
        "expected_error": "invalid_arguments",
    }


def with_action(batch, **fields):
    updated = deepcopy(batch)
    updated["arguments"].update(fields)
    return updated


LEADS = (
    turn("For the scanner export card, what job does input_file_extension do?",
         call("explain_purpose", subject="input_file_extension")),
    turn("Before the configuration walkthrough, describe the role of slice_time_ref.",
         call("explain_purpose", subject="slice_time_ref")),
    turn("Open the fitting handoff with what nDCT is used for.",
         call("explain_purpose", subject="nDCT")),
    turn("Start this onboarding card by explaining the function of batches.",
         call("explain_purpose", subject="batches")),
)

COMPANIONS = (
    ("separately explain what results.basepath is used for", call("explain_purpose", subject="results.basepath")),
    ("separately describe the purpose of input_file_extension", call("explain_purpose", subject="input_file_extension")),
    ("separately tell me what optional_params.filepath does", call("explain_purpose", subject="optional_params.filepath")),
    ("separately explain the role of results_anaylsis_id", call("explain_purpose", subject="results_anaylsis_id")),
)

DECOYS = (
    turn("For a different line on the card, describe the purpose of individual.run.",
         call("explain_purpose", subject="individual.run")),
    turn("In a separate note, what function does measured_data_filepath serve?",
         call("explain_purpose", subject="measured_data_filepath")),
    turn("Before we finish the checklist, what use does hrf.key have?",
         call("explain_purpose", subject="hrf.key")),
    turn("One separate item for the margin: explain the role of space.",
         call("explain_purpose", subject="space")),
)

ORDINALS = ("second", "third", "fourth")


def prelude(index):
    count = index % len(ORDINALS) + 1
    return [LEADS[(index + offset) % len(LEADS)] for offset in range(count)]


REPEAT = (
    ("sampling-card", "what units distinguish the visual_field_radius setting", "get_units", "visual_field_radius",
     "what the radius setting controls", "explain_purpose",
     "Copy the scientific wording of the radius answer in item {part} of your reply to my {ordinal} request, as it stood before its correction. If that item never appeared, ask which answer I intend."),
    ("kernel-handout", "which convention gives the units for peak_delay", "get_units", "peak_delay",
     "the documented default for peak_delay", "get_default_value",
     "For the handout, take part {part} of your {ordinal} reply and reproduce its original peak_delay answer verbatim. Ask which answer if the scientific part is missing."),
    ("file-catalogue", "what stimulus_filepath is intended to control", "explain_purpose", "stimulus_filepath",
     "what contents belong in stimulus_filepath", "describe_path_target",
     "My catalogue needs the original stimulus_filepath wording from answer part {part} to my {ordinal} message. If no answer occupies that slot, ask me to identify it."),
    ("site-handoff", "what BIDS.basepath is for", "explain_purpose", "BIDS.basepath",
     "which data should live under BIDS.basepath", "describe_path_target",
     "Paste the BIDS.basepath answer from section {part} of your {ordinal} reply as first delivered. If you did not give a scientific answer in that section, ask what I want copied."),
    ("hemisphere-cards", "which settings are supported for hemi", "get_allowed_values", "hemi",
     "whether hemi matches letter case", "explain_case_sensitivity",
     "I want the uncorrected hemi text from item {part} of your answer to my {ordinal} request. Reproduce just that item, or ask which answer if it was never shown."),
    ("grid-ledger", "what the width parameter controls", "explain_purpose", "width",
     "what unit goes with width", "get_units",
     "Restore the exact width answer as first printed in slot {part} of your response to my {ordinal} request. Ask me which answer to restore if that slot was never filled."),
    ("gpu-workbook", "how GEM-pRF uses additional_available_gpus", "explain_purpose", "additional_available_gpus",
     "what default is documented for additional_available_gpus", "get_default_value",
     "For the GPU workbook, use only the text originally shown as item {part} when I asked my {ordinal} question. Ask which answer I mean if you never showed that item."),
    ("template-record", "what the version setting is for", "explain_purpose", "version",
     "what XML version string format the loader requires", "explain_version_format",
     "Transcribe the earlier version answer that first occupied part {part} of your {ordinal} reply. If part {part} did not get an answer, ask me which response to transcribe."),
)

REPLACE = (
    ("grid-columns", "what num_horizontal_prfs controls", "explain_purpose", "num_horizontal_prfs",
     "how the default horizontal grid includes the zero position depending on odd or even count",
     call("explain_grid_axis_geometry", subject="num_horizontal_prfs", property="zero_inclusion")),
    ("smoothing-numbers", "what min_sigma controls", "explain_purpose", "min_sigma",
     "the documented units of min_sigma", call("get_units", subject="min_sigma")),
    ("axis-units", "what unit height has", "get_units", "height",
     "the units of width", call("get_units", subject="width")),
    ("naming-record", "what custom_filename_postfix is for", "explain_purpose", "custom_filename_postfix",
     "the naming convention for custom_filename_postfix", call("explain_filename_format", subject="custom_filename_postfix")),
    ("label-domain", "what input_file_extension is for", "explain_purpose", "input_file_extension",
     "the value domain supported for input_file_extension", call("get_allowed_values", subject="input_file_extension")),
    ("subject-selection", "what individual.ses is for", "explain_purpose", "individual.ses",
     "what individual.run is for", call("explain_purpose", subject="individual.run")),
    ("optional-table", "what optional_params.filepath is for", "explain_purpose", "optional_params.filepath",
     "what should be stored at optional_params.filepath",
     call("describe_path_target", subject="optional_params.filepath", aspect="intended_contents")),
    ("setting-convention", "what prepend_date controls", "explain_purpose", "prepend_date",
     "whether case matters for prepend_date", call("explain_case_sensitivity", subject="prepend_date")),
)

REPLACE_QUESTIONS = (
    "Replace item {part} in the answer to my {ordinal} request with {meaning}, keeping its other item. If there was no item {part}, give the requested answer anew. If it already says that, repeat that item.",
    "Your {ordinal} reply goes in my record. Swap out just item {part} for {meaning}. Provide the new answer alone if that slot is empty, or copy it if already correct.",
    "Update section {part} of your {ordinal} reply to cover {meaning}; leave the neighbouring section alone. Answer from scratch if the earlier section is absent. Copy it if equivalent.",
    "For my correction log, substitute {meaning} for part {part} of your {ordinal} reply, leaving the other answer intact. Give it fresh if that slot has no answer. Repeat it if it already matches.",
)

FLAGS = (
    ("refinement", "refine_fitting.enable", "normalize", True),
    ("grid-input", "spatial_grid_xy.use_from_file", "hrf.use_from_file", False),
    ("sigma-input", "sigmas.use_from_file", "refinefit_on_gpu", True),
    ("debug-output", "write_debug_info", "results_anaylsis_id.overwrite", False),
    ("time-resolution", "high_temporal_resolution.enable", "optional_params.enable", True),
)


def clarification(question, name, **known):
    return turn(question, call(name, **known, _clarify=True))


def pending_specs():
    for name, subject, decoy, value in FLAGS:
        setting = "enabled" if value else "disabled"
        other = "disabled" if value else "enabled"
        target = clarification(
            f"I want to know the effect of my {subject} choice. I haven't told you whether it is on.",
            "explain_boolean_effect", subject=subject,
        )
        distractor = clarification(
            f"Open a separate question about what my {decoy} choice does; I will give you that setting later.",
            "explain_boolean_effect", subject=decoy,
        )
        finish = call("explain_boolean_effect", subject=subject, value=value)
        before = call("explain_boolean_effect", subject=subject, value=not value)
        new = (f"For the unfinished {subject} answer from my {{ordinal}} request, the flag is {setting}. Close only that "
               f"request, leaving the {decoy} question open. If that request is no longer open, explain the "
               f"{setting} setting as a new answer.")
        old = f"For the earlier {subject} question, use {other} for now. Leave the other question open."
        yield name, target, distractor, finish, before, {"value": value}, new, old

    yield (
        "units-card",
        clarification("I need a note about a time parameter's units. Ask me which parameter I am filing.", "get_units"),
        clarification("Keep a second request open about units for a spatial parameter. I'll name it another time.",
                      "get_units"),
        call("get_units", subject="peak_delay"),
        call("get_units", subject="under_shoot_delay"),
        {"subject": "peak_delay"},
        "On my {ordinal} request, the parameter is peak_delay. Supply its units and leave the separate spatial-units question open. If that request is closed, answer peak_delay as a fresh units question.",
        "Use under_shoot_delay for the earlier time-units inquiry and leave the other question pending.",
    )
    yield (
        "path-aspect",
        clarification("For stimulus.directory I have not decided which path detail to check. Ask which aspect.",
                      "describe_path_target", subject="stimulus.directory"),
        clarification("Hold a separate inquiry about measured_data_filepath until I specify the aspect.",
                      "describe_path_target", subject="measured_data_filepath"),
        call("describe_path_target", subject="stimulus.directory", aspect="filesystem_object_kind"),
        call("describe_path_target", subject="stimulus.directory", aspect="intended_contents"),
        {"aspect": "filesystem_object_kind"},
        "For my {ordinal} question, just tell me whether stimulus.directory is a file or a directory. Leave the measured_data_filepath inquiry pending. If that question is no longer open, give the path-kind answer fresh here.",
        "Close the earlier stimulus.directory inquiry with what contents belong there. Keep the second path question open.",
    )
    yield (
        "default-card",
        clarification("I need a documented configuration default for the HRF card. Ask me to identify the parameter.",
                      "get_default_value"),
        clarification("Open a separate default-value inquiry for the grid card, but wait for its parameter name.",
                      "get_default_value"),
        call("get_default_value", subject="under_shoot_delay"),
        call("get_default_value", subject="peak_delay"),
        {"subject": "under_shoot_delay"},
        "For my {ordinal} request use under_shoot_delay and tell me its documented default. Keep the grid default question open. If the HRF request isn't pending, answer under_shoot_delay fresh.",
        "Use peak_delay for the earlier HRF default inquiry. Keep the other default request pending.",
    )


def example(group, family, variant, history, question, policy, targets):
    family_id = f"targeted-train-{group}-{family}"
    return {
        "case_id": f"{family_id}-{variant}",
        "family_id": family_id,
        "group": group,
        "history": history,
        "question": question,
        "policy": policy,
        "expected_targets": targets,
    }


def original_call(operation, subject):
    if operation == "describe_path_target":
        return call(operation, subject=subject, aspect="intended_contents")
    return call(operation, subject=subject)


def compound(part, question, primary, i):
    other, decoy = COMPANIONS[i % len(COMPANIONS)]
    if part == 1:
        return turn(f"Please explain {question}; {other}.", primary, decoy)
    lower = other.removeprefix("separately ")
    return turn(f"First {lower}; then explain {question}.", decoy, primary)


def repeat_cases(core):
    for i, (family, aim, operation, subject, edit, edit_op, prompt) in enumerate(REPEAT):
        part = i % 2 + 1
        leads, decoy = prelude(i), DECOYS[i % len(DECOYS)]
        ordinal = ORDINALS[len(leads) - 1]
        exchange = len(leads) + 1
        opening = compound(part, aim, call(operation, subject=subject), i)
        edit_call = original_call(edit_op, subject)
        edit_question = (
            f"On the {subject} item in your {ordinal} reply, I now want {edit}. "
            "Change just that answer if it exists, and give the new explanation directly if no answer appeared."
        )
        question = prompt.format(part=part, ordinal=ordinal)
        policy = {"kind": "repeat_answer", "target_exchange": exchange, "part": part}
        for variant, first, second, distractor, target_turn in (
            ("clean", leads[0], opening, decoy, exchange),
            ("lead-error", failed(core, leads[0]), opening, decoy, exchange - 1),
            ("target-absent", leads[0], failed(core, opening), decoy, None),
            ("intervening-error", leads[0], opening, failed(core, decoy), exchange),
        ):
            target = f"t{target_turn}.c{part}" if target_turn else None
            revision = with_action(edit_call, _replace=target) if target else edit_call
            history = [first, *leads[1:], second, distractor, turn(edit_question, revision)]
            yield example("repeat_original", family, variant, history, question, policy, [target] if target else [])


def replacement_cases(core):
    for i, (family, aim, operation, subject, meaning, replacement) in enumerate(REPLACE):
        part = (i + 1) % 2 + 1
        leads, decoy = prelude(i + 1), DECOYS[(i + 2) % len(DECOYS)]
        ordinal = ORDINALS[len(leads) - 1]
        exchange = len(leads) + 1
        opening = compound(part, aim, call(operation, subject=subject), i)
        already = compound(part, meaning, replacement, i)
        question = REPLACE_QUESTIONS[i % len(REPLACE_QUESTIONS)].format(part=part, ordinal=ordinal, meaning=meaning)
        policy = {"kind": "replace", "target_exchange": exchange, "part": part, "calls": [replacement]}
        for variant, first, second, target_turn in (
            ("clean", leads[0], opening, exchange),
            ("lead-error", failed(core, leads[0]), opening, exchange - 1),
            ("target-absent", leads[0], failed(core, opening), None),
            ("already-correct", leads[0], already, exchange),
        ):
            targets = [f"t{target_turn}.c{part}"] if target_turn else []
            yield example("replace_answer", family, variant, [first, *leads[1:], second, decoy], question, policy, targets)


def resolution_cases(core):
    for i, (family, target, decoy, finish, old, missing, question, old_question) in enumerate(pending_specs()):
        leads = prelude(i + 2)
        question = question.format(ordinal=ORDINALS[len(leads) - 1])
        exchange = len(leads) + 1
        policy = {
            "kind": "pending_resolve",
            "target_exchange": exchange,
            "function": finish["name"],
            "values": missing,
            "fallback_calls": [finish],
        }
        for variant, first, second, target_turn in (
            ("clean", leads[0], target, exchange),
            ("lead-error", failed(core, leads[0]), target, exchange - 1),
            ("target-absent", leads[0], failed(core, target), None),
            ("previously-resolved", leads[0], target, None),
        ):
            history = [first, *leads[1:], second, decoy]
            if variant == "previously-resolved":
                values = {key: old["arguments"][key] for key in missing}
                history.append(turn(old_question, call(old["name"], **values, _resolve=f"t{exchange}.p1")))
            targets = [f"t{target_turn}.p1"] if target_turn else []
            yield example("resolve_pending", family, variant, history, question, policy, targets)


def validate(registry, core, harness, row):
    state = core.createSession(registry, {"sessionId": row["case_id"], "dialogueVersion": 2})
    for step in row["history"]:
        raw = step["raw_output"] if "raw_output" in step else core.encodeCalls(step["calls"])
        state = core.applyModelOutput(registry, state, step["question"], raw)
        last = state["exchanges"][-1]
        if last["outcome"] != step["expected_outcome"] or last.get("error_code") != step.get("expected_error"):
            raise ValueError(f"Unexpected history outcome {last.get('error_code')}: {row['case_id']} {step['question']}")
    resolved = harness.resolve(row, state)
    targets = row["expected_targets"]
    if resolved["reference_targets"] != targets:
        raise ValueError(f"Wrong authored reference target: {row['case_id']}")
    batch = resolved["allowed_batches"][0]
    after = core.applyModelOutput(registry, state, row["question"], core.encodeCalls(batch))
    last = after["exchanges"][-1]
    if last["outcome"] != "committed":
        raise ValueError(f"Authored continuation fails with {last.get('error_code')}: {row['case_id']}")

    before_pending = state["semantic"]["pending"]
    resolves_pending = row["group"] == "resolve_pending"
    removed = targets if resolves_pending else []
    remaining = [entry for entry in before_pending if entry["id"] not in removed]
    if (
        after["exchanges"][:-1] != state["exchanges"]
        or after["semantic"]["turns"][:-1] != state["semantic"]["turns"]
        or after["semantic"]["pending"] != remaining
    ):
        raise ValueError(f"Authored continuation changes unrelated history: {row['case_id']}")
    if resolves_pending and len(before_pending) != (2 if targets else 1):
        raise ValueError(f"Authored continuation lacks the expected competing clarification: {row['case_id']}")
    linked = [
        (relation, answer[relation])
        for answer in after["semantic"]["turns"][-1]["answers"]
        for relation in ("repeats", "revises", "resolves")
        if relation in answer
    ]
    relation = "resolves" if resolves_pending else "revises"
    if row["group"] == "repeat_original" or row["case_id"].endswith("-already-correct"):
        relation = "repeats"
    if linked != [(relation, target) for target in targets]:
        raise ValueError(f"Authored continuation has the wrong history effect: {row['case_id']}")


def cases(registry):
    core = api()
    harness = createActualHistoryHarness(core)
    for source in repeat_cases, replacement_cases, resolution_cases:
        for row in source(core):
            validate(registry, core, harness, row)
            yield row
