"""Independently authored diagnostics for dialogue references and function arguments."""

from functools import partial

from evaluation.functiongemma import encodeCalls


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _subject(name, subject):
    return _call(name, subject=subject)


def _boolean(subject, value):
    return _call("explain_boolean_effect", subject=subject, value=value)


def _clarify_value(subject):
    return _call("explain_boolean_effect", subject=subject, _clarify=True, _missing=["value"])


def _fresh(question, *calls):
    return {"question": question, "policy": {"kind": "fresh", "calls": list(calls)}, "expected_targets": []}


def _reference(question, kind, exchange, target, **policy):
    return {
        "question": question,
        "policy": {"kind": kind, "target_exchange": exchange, **policy},
        "expected_targets": [target] if target else [],
    }


def _replace(question, exchange, target, part, *calls):
    return _reference(question, "replace", exchange, target, part=part, calls=list(calls))


def _resolve(question, exchange, target, subject, value):
    return _reference(
        question, "pending_resolve", exchange, target, function="explain_boolean_effect",
        values={"value": value}, fallback_calls=[_boolean(subject, value)],
    )


def _episodes():
    unit = partial(_subject, "get_units")
    purpose = partial(_subject, "explain_purpose")
    default = partial(_subject, "get_default_value")
    options = partial(_subject, "get_allowed_values")
    use = partial(_subject, "explain_code_usage")

    yield "notebook-timing", "repeat_original", [
        _fresh("My lab notebook needs two separate entries. What unit goes with TR? What job does slice_time_ref do?",
               unit("TR"), purpose("slice_time_ref")),
        _fresh("For the next page, enumerate the supported input_file_extension values.", options("input_file_extension")),
        _replace("Edit just the TR entry in your first reply to say what GEM-pRF supplies as its default. If that entry was never made, give the default on its own.",
                 1, "t1.c1", 1, default("TR")),
        _fresh("New topic for my notebook. What is num_frames_downsampled used for?", purpose("num_frames_downsampled")),
        _reference("I'm comparing notebook revisions. Read back the unedited TR unit answer from reply one. If you can't identify a scientific answer there, ask which answer I mean.",
                   "repeat_answer", 1, "t1.c1", part=1),
    ]

    yield "archive-path", "repeat_original", [
        _fresh("In order, tell me what measured_data_filepath is meant to contain and what concatenate_item controls.",
               _call("describe_path_target", subject="measured_data_filepath", aspect="intended_contents"),
               purpose("concatenate_item")),
        _fresh("I also need the list of legal run_type settings for the archive.", options("run_type")),
        _replace("Change the first subanswer of your opening reply to discuss how portable measured_data_filepath is between computers. If no subanswer exists, answer that portability question on its own.",
                 1, "t1.c1", 1, _subject("explain_path_portability", "measured_data_filepath")),
        _fresh("For a separate table, give the unit convention associated with default_hrf.t.", unit("default_hrf.t")),
        _reference("For the archive, quote just the answer about intended measured-data contents as it stood when you first sent it. If you cannot locate that answer, ask me to identify it.",
                   "repeat_answer", 1, "t1.c1", part=1),
    ]

    yield "registration-notes", "repeat_original", [
        _fresh("Make a two-item reference card. List valid hemi values first and explain what the analysis option identifies second.",
               options("hemi"), purpose("analysis")),
        _fresh("What value can I document as the default, if any, for the space setting?", default("space")),
        _replace("In your initial reference card, edit item two to point to the code that uses analysis. If there was no second item, just tell me where the code uses it.",
                 1, "t1.c2", 2, use("analysis")),
        _fresh("Write a separate one-line explanation of what individual.ses does.", purpose("individual.ses")),
        _reference("I'm auditing the change to item two. Return its initial explanation of what analysis identifies. If the original subanswer can't be identified, please clarify.",
                   "repeat_answer", 1, "t1.c2", part=2),
    ]

    yield "grid-entry", "replace_answer", [
        _fresh("For an unrelated heading, what is visual_field_radius for?", purpose("visual_field_radius")),
        _fresh("In order, what units are used for width, and how does num_horizontal_prfs affect spacing?",
               unit("width"), _call("explain_grid_axis_geometry", subject="num_horizontal_prfs", property="spacing")),
        _fresh("I also need the code location that reads num_horizontal_prfs.", use("num_horizontal_prfs")),
        _replace("Edit only your spacing answer in the second reply to address whether zero is included along the same axis. If that subanswer is absent, answer the zero question on its own.",
                 2, "t2.c2", 2,
                 _call("explain_grid_axis_geometry", subject="num_horizontal_prfs", property="zero_inclusion")),
        _reference("Verify my notes by reading back the width-units subanswer from reply two. If it was never recorded, ask which answer I want.",
                   "repeat_answer", 2, "t2.c1", part=1),
    ]

    yield "results-entry", "replace_answer", [
        _fresh("First document the purpose of the results directory setting results.basepath.", purpose("results.basepath")),
        _fresh("Make two entries in this order. What is the default for results_anaylsis_id.overwrite? What does prepend_date do?",
               default("results_anaylsis_id.overwrite"), purpose("prepend_date")),
        _fresh("Separately, what code reads the setting for custom_filename_postfix?", use("custom_filename_postfix")),
        _replace("Change the second entry from reply two so it explains what literal False for prepend_date does. If you cannot find that entry, answer just what the literal does.",
                 2, "t2.c2", 2, _call("explain_flag_literal_effect", subject="prepend_date", value="False")),
        _reference("Read exactly the overwrite-default part from that two-entry reply. Ask me to locate it if it is missing.",
                   "repeat_answer", 2, "t2.c1", part=1),
    ]

    yield "condition-entry", "replace_answer", [
        _fresh("What purpose does the optional_params.filepath setting serve?", purpose("optional_params.filepath")),
        _fresh("Please answer in order. Where is sigmas.key used in code? What is the default setting for hrf.use_from_file?",
               use("sigmas.key"), default("hrf.use_from_file")),
        _fresh("For a separate checklist, what type of filesystem object is stimulus.directory intended to name?",
               _call("describe_path_target", subject="stimulus.directory", aspect="filesystem_object_kind")),
        _replace("Amend the HRF default entry in your second reply to describe the effect when hrf.use_from_file is true. If that entry is unavailable, give that effect as a separate answer.",
                 2, "t2.c2", 2, _boolean("hrf.use_from_file", True)),
        _reference("Return the sigmas.key code-usage subanswer from your second reply exactly as shown, or clarify if that entry is missing.",
                   "repeat_answer", 2, "t2.c1", part=1),
    ]

    pending_episodes = (
        ("loading-choices", "hrf.use_from_file", True, "sigmas.use_from_file", False,
         "Between those items, tell me where default_hrf.t is read in code.", use("default_hrf.t")),
        ("fitting-choices", "refinefit_on_gpu", False, "normalize", True,
         "As a separate item for the fitting sheet, what is max_sigma measured in?", unit("max_sigma")),
        ("runtime-choices", "write_debug_info", True, "optional_params.enable", False,
         "One separate detail for this checklist: what XML version is documented by default?", default("version")),
    )
    for slug, first, left, second, right, other_question, other_call in pending_episodes:
        yield slug, "resolve_pending", [
            _fresh(f"Open an item on how my {first} setting behaves. I still need to look up whether it's enabled, so ask me for its value.",
                   _clarify_value(first)),
            _fresh(f"Open another item about the effect of my {second} setting. Its value is missing from my notes too, so ask separately.",
                   _clarify_value(second)),
            _fresh(other_question, other_call),
            _resolve(f"Finish the open {second} item. It is {str(right).lower()}. Leave the other item open. If this item never opened, answer its effect alone.",
                     2, "t2.p1", second, right),
            _resolve(f"Now finish the {first} item that I raised before it. It is {str(left).lower()}. If that item never opened, explain this effect on its own.",
                     1, "t1.p1", first, left),
        ]


def _committed(question, *calls):
    return {"question": question, "calls": list(calls), "expected_outcome": "committed"}


def _failed(step):
    return {
        "question": step["question"],
        "raw_output": encodeCalls([_call("targeted_dev_unavailable_function")]),
        "expected_outcome": "error",
    }


def _contrast_families():
    examples = (
        ("radius-quote", "repeat_original",
         _committed("For a figure caption, what does visual_field set?", _subject("explain_purpose", "visual_field")),
         _committed("Write my methods units in two parts: first min_sigma, then visual_field_radius.",
                    _subject("get_units", "min_sigma"), _subject("get_units", "visual_field_radius")),
         _committed("One unrelated axis setting: what is the documented role of num_vertical_prfs?",
                    _subject("explain_purpose", "num_vertical_prfs")),
         "Copy only the radius-units part from the methods reply. If you never supplied it, ask which answer to copy.",
         {"kind": "repeat_answer", "target_exchange": 2, "part": 2}, "c2"),
        ("labels-quote", "repeat_original",
         _committed("Explain the purpose of individual.run for my bookkeeping.",
                    _subject("explain_purpose", "individual.run")),
         _committed("Give the documented options for space and then for slice_time_ref, separately.",
                    _subject("get_allowed_values", "space"), _subject("get_allowed_values", "slice_time_ref")),
         _committed("For a different note, provide the default for individual.ses.",
                    _subject("get_default_value", "individual.ses")),
         "Fill in my registration note by copying the space options from the two-answer reply. If that scientific answer is absent, ask me to identify it.",
         {"kind": "repeat_answer", "target_exchange": 2, "part": 1}, "c1"),
        ("axis-replace", "replace_answer",
         _committed("For the legend, define the role of height.", _subject("explain_purpose", "height")),
         _committed("On my axis card, report the unit of height first and vertical-axis spacing second.",
                    _subject("get_units", "height"),
                    _call("explain_grid_axis_geometry", subject="num_vertical_prfs", property="spacing")),
         _committed("I also want to know where width is processed in code.", _subject("explain_code_usage", "width")),
         "Edit just the vertical-spacing answer on the axis card to discuss zero inclusion on that axis. If the prior subanswer is missing, answer zero inclusion alone.",
         {"kind": "replace", "target_exchange": 2, "part": 2,
          "calls": [_call("explain_grid_axis_geometry", subject="num_vertical_prfs", property="zero_inclusion")]}, "c2"),
        ("path-replace", "replace_answer",
         _committed("What is the purpose of results_anaylsis_id?", _subject("explain_purpose", "results_anaylsis_id")),
         _committed("On my setup card put the meaning of BIDS.basepath first and the file type accepted by stimulus.directory second.",
                    _subject("explain_purpose", "BIDS.basepath"),
                    _call("describe_path_target", subject="stimulus.directory", aspect="filesystem_object_kind")),
         _committed("New topic: what role does stimulus_filepath have?", _subject("explain_purpose", "stimulus_filepath")),
         "Amend only the BIDS.basepath part of the setup card to explain portability between machines. If that answer didn't appear, explain portability on its own.",
         {"kind": "replace", "target_exchange": 2, "part": 1,
          "calls": [_subject("explain_path_portability", "BIDS.basepath")]}, "c1"),
        ("filter-pending", "resolve_pending",
         _committed("What does the hemi setting represent?", _subject("explain_purpose", "hemi")),
         _committed("Ask me for refine_fitting.enable before explaining what my setting changes.",
                    _clarify_value("refine_fitting.enable")),
         _committed("Keep a separate unanswered item about hrf.use_from_file. You'll have to ask for its value too.",
                    _clarify_value("hrf.use_from_file")),
         "For the refine_fitting.enable question I've left open, the value is false. Leave the HRF item alone. Give the fitting effect by itself if this question never opened.",
         {"kind": "pending_resolve", "target_exchange": 2, "function": "explain_boolean_effect",
          "values": {"value": False}, "fallback_calls": [_boolean("refine_fitting.enable", False)]}, "p1"),
        ("sigma-pending", "resolve_pending",
         _committed("For the first line, what does num_sigmas control?", _subject("explain_purpose", "num_sigmas")),
         _committed("The effect depends on whether sigmas.use_from_file is on in my setup. Please ask for that setting first.",
                    _clarify_value("sigmas.use_from_file")),
         _committed("Separately ask for the value of normalize before you explain that effect.", _clarify_value("normalize")),
         "The value I withheld for sigmas.use_from_file is true. Resolve that item and keep normalize pending. If the sigma item is absent, explain its effect alone.",
         {"kind": "pending_resolve", "target_exchange": 2, "function": "explain_boolean_effect",
          "values": {"value": True}, "fallback_calls": [_boolean("sigmas.use_from_file", True)]}, "p1"),
    )
    for slug, group, primer, addressed, distractor, question, policy, part in examples:
        histories = (
            ("earlier_pass", [primer, addressed, distractor], [f"t2.{part}"]),
            ("earlier_error", [_failed(primer), addressed, distractor], [f"t1.{part}"]),
            ("target_unavailable", [primer, _failed(addressed), distractor], []),
        )
        for variant, history, targets in histories:
            yield slug, group, variant, {
                "history": history, "question": question, "policy": policy,
                "expected_targets": targets,
            }


def _argument_cases():
    yield "width-data", _fresh(
        "I'm cross-checking a stimulus-loading bug. Which on-disk stimulus dimension should equal the configured width?",
        _call("explain_parameter_data_matching", subject="width", reference="stimulus_file_width"),
    )
    yield "height-data", _fresh(
        "For that kind of stimulus-file cross-check, which dimension is expected to match the XML height field?",
        _call("explain_parameter_data_matching", subject="height", reference="stimulus_file_height"),
    )
    yield "horizontal-zero", _fresh(
        "Does adjusting the horizontal candidate count change whether zero eccentricity is represented along that axis?",
        _call("explain_grid_axis_geometry", subject="num_horizontal_prfs", property="zero_inclusion"),
    )
    yield "date-false-literal", _fresh(
        "My config literally contains False for prepend_date. How does the loader interpret that spelling?",
        _call("explain_flag_literal_effect", subject="prepend_date", value="False"),
    )
    yield "version-omitted", _fresh(
        "My XML root has no version attribute at all. What happens when GEM-pRF reads it?",
        _call("explain_unset_value", subject="version", value_state="omitted"),
    )
    yield "hemi-labels", _fresh(
        "For the hemisphere filter, I'm typing the l and r labels. Would changing their case select the same hemispheres?",
        _call("explain_label_case_acceptance", subject="hemi", literals=["l", "r"]),
    )


def _row(family, key, group, panel, turn, content):
    return {
        "case_id": f"{family}-{key}",
        "family_id": family,
        "bootstrap_family_id": family,
        "group": group,
        "panel": panel,
        "split": "dev",
        "cohort": "supported",
        "turn_index": turn,
        "reference_eligible": content["policy"]["kind"] != "fresh",
        **content,
    }


def cases(registry):
    """Yield public DEV declarations in episode and turn order."""
    for slug, group, dialogue in _episodes():
        family = f"targeted-dev-episode-{slug}"
        for index, content in enumerate(dialogue, 1):
            row = _row(family, f"t{index}", group, "closed_loop", index, content)
            yield {**row, "dialogue_id": family}

    for slug, group, variant, content in _contrast_families():
        family = f"targeted-dev-challenge-{slug}"
        yield _row(family, variant, group, "challenge", 1, content)

    for slug, content in _argument_cases():
        family = f"targeted-dev-arguments-{slug}"
        yield _row(family, "case", "function_arguments", "challenge", 1, {"history": [], **content})
