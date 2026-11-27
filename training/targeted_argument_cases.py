"""New TRAIN tasks for function selection and argument validation."""


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def examples():
    yield (
        "density",
        "Refinement stays on in this analysis. Can the source justify a generally sufficient search-grid density under that setting?",
        [call("explain_grid_density_guidance", refinement_enabled=True)],
        call("get_value_guidance", subject="grid_density"),
    )
    yield (
        "density",
        "This fit will skip refinement. What limits does the code place on a recommendation for the search-grid density?",
        [call("explain_grid_density_guidance", refinement_enabled=False)],
        call("get_value_guidance", subject="grid_density"),
    )
    yield (
        "density",
        'The XML literally compares the refinement text against "True" and takes the unequal branch. Explain what that means for the grid density guidance.',
        [call("explain_grid_density_under_xml_condition", operator="not_equal", literal="True")],
        None,
    )

    for axis, subject in (("horizontal", "num_horizontal_prfs"), ("vertical", "num_vertical_prfs")):
        yield (
            "axis",
            f"I need to describe the nominal {axis} spacing from the count {subject}. What does the implementation establish about that spacing?",
            [call("explain_grid_axis_geometry", subject=subject, property="spacing")],
            call("explain_grid_axis_geometry", subject=subject, property=False),
        )
        yield (
            "axis",
            f"For an odd default count on the {axis} axis ({subject}), does the coordinate grid include zero? Explain that property of the generated axis.",
            [call("explain_grid_axis_geometry", subject=subject, property="zero_inclusion")],
            call("explain_grid_axis_geometry", subject=subject, property=False),
        )
    yield (
        "axis",
        "One colleague wants the horizontal spacing and another wants to know whether the odd vertical count includes zero. Please give one answer per request in that order.",
        [
            call("explain_grid_axis_geometry", subject="num_horizontal_prfs", property="spacing"),
            call("explain_grid_axis_geometry", subject="num_vertical_prfs", property="zero_inclusion"),
        ],
        call("explain_grid_axis_geometry", subject="num_vertical_prfs", property=False),
    )

    for subject in ("spatial_grid_xy.use_from_file", "hrf.use_from_file", "sigmas.use_from_file"):
        for enabled in (False, True):
            yield (
                "file_flag",
                f"My setup has {subject} {'enabled' if enabled else 'disabled'}. Explain the behavior of this switch for that setting.",
                [call("explain_boolean_effect", subject=subject, value=enabled)],
                call("explain_flag_literal_effect", subject=subject, value="True" if enabled else "False"),
            )
    for subject in ("BIDS.enable", "binarization.enable", "prepend_date"):
        for literal in ("False", "True"):
            yield (
                "xml_flag",
                f'The XML value of {subject} is exactly "{literal}". Which behavior follows when the loader sees that spelling?',
                [call("explain_flag_literal_effect", subject=subject, value=literal)],
                call("explain_flag_literal_effect", subject=subject, value=literal == "True"),
            )

    yield (
        "empty_value",
        "I kept the TR field in the XML and left the value blank. How does GEM-pRF handle that empty field?",
        [call("explain_unset_value", subject="TR", value_state="empty")],
        call("explain_unset_value", subject="min_sigma", value_state="empty"),
    )
    yield (
        "empty_value",
        "This copy of the XML lacks the TR field entirely. What is the documented handling of an omitted TR?",
        [call("explain_unset_value", subject="TR", value_state="omitted")],
        call("explain_unset_value", subject="HRF", value_state="omitted"),
    )
    yield (
        "empty_value",
        "If the user removes the XML version attribute altogether, what does this revision do on load?",
        [call("explain_unset_value", subject="version", value_state="omitted")],
        None,
    )
    yield (
        "empty_value",
        "The output naming field custom_filename_postfix is present but contains no text. What happens with this empty setting?",
        [call("explain_unset_value", subject="custom_filename_postfix", value_state="empty")],
        None,
    )
    yield (
        "empty_value",
        "With only one GPU available, what happens if additional_available_gpus is present and empty?",
        [
            call(
                "explain_unset_value",
                subject="additional_available_gpus",
                value_state="empty",
                available_gpu_count=1,
            )
        ],
        None,
    )

    yield (
        "supplied_value",
        'I want to use every task and wrote the literal "all" in individual.task. Does the source admit that value?',
        [call("explain_value_acceptance", subject="individual.task", value="all")],
        call("explain_value_acceptance", subject="all", value="all"),
    )
    yield (
        "supplied_value",
        "The candidate value for min_sigma is numeric zero. Can I use it as supplied according to the implementation?",
        [call("explain_value_acceptance", subject="min_sigma", value=0)],
        call("explain_unset_value", subject="min_sigma", value_state="empty"),
    )
    yield (
        "supplied_value",
        "I am testing a single sampled sigma by entering the number one for num_sigmas. Is this value allowed?",
        [call("explain_value_acceptance", subject="num_sigmas", value=1)],
        None,
    )

    for subject, reference, wording in (
        ("TR", "fmri_acquisition_repetition_time", "the repetition time actually used to acquire my fMRI data"),
        ("height", "stimulus_file_height", "the image height in the stored stimulus file"),
        ("width", "stimulus_file_width", "the image width in the stored stimulus file"),
        ("individual.task", "stimulus_filename_task_entity", "the task entity in the stimulus filename"),
        ("num_frames_downsampled", "measured_timepoint_count", "the count of measured timepoints"),
    ):
        yield (
            "data_matching",
            f"I am cross-checking configured {subject} against {wording}. What relationship between those two quantities does GEM-pRF expect?",
            [call("explain_parameter_data_matching", subject=subject, reference=reference)],
            call("explain_parameter_data_matching", subject=subject, reference="hemi_acquisition_repetition_time"),
        )
    yield (
        "data_matching",
        "I have two metadata checks. First compare configured width with the stimulus file's width. Then compare configured TR with the fMRI acquisition repetition time.",
        [
            call("explain_parameter_data_matching", subject="width", reference="stimulus_file_width"),
            call("explain_parameter_data_matching", subject="TR", reference="fmri_acquisition_repetition_time"),
        ],
        call("explain_parameter_data_matching", subject="hemi", reference="fmri_acquisition_repetition_time"),
    )
    yield (
        "data_matching",
        "For one input, check both axes separately: does configured height need to match the stimulus file height, and does configured width need to match the stimulus file width?",
        [
            call("explain_parameter_data_matching", subject="height", reference="stimulus_file_height"),
            call("explain_parameter_data_matching", subject="width", reference="stimulus_file_width"),
        ],
        None,
    )

    yield (
        "hemisphere",
        "Can the hemi filename filter apply if my measured input is a volumetric .nii.gz file? Limit the answer to that filter in this representation.",
        [call("explain_hemisphere_filter_applicability", subject="hemi", file_extension=".nii.gz")],
        call(
            "explain_hemisphere_filter_applicability", subject="custom_filename_postfix", file_extension=".nii.gz"
        ),
    )
    yield (
        "hemisphere",
        "For the hemi filename filter, are lowercase l and r the accepted labels, and is their case significant?",
        [call("explain_label_case_acceptance", subject="hemi", literals=["l", "r"])],
        call("explain_label_case_acceptance", subject="BIDS.enable", literals=["l", "r"]),
    )
    yield (
        "hemisphere",
        "Answer two separate questions about hemi. Explain the case treatment of the pair l and r, then explain whether the filter applies to .nii.gz volumes.",
        [
            call("explain_label_case_acceptance", subject="hemi", literals=["l", "r"]),
            call("explain_hemisphere_filter_applicability", subject="hemi", file_extension=".nii.gz"),
        ],
        None,
    )

    yield (
        "labels",
        'My analysis selector is written "01". Compare that exact label with "analysis-01" for the analysis setting.',
        [call("compare_label_literals", subject="analysis", left="01", right="analysis-01")],
        call("compare_label_literals", subject="01", left="01", right="analysis-01"),
    )
    yield (
        "labels",
        'In the subject selector, how does the exact spelling "01" compare with "sub-01"? Keep the prefixes and leading zero in the explanation.',
        [call("compare_label_literals", subject="sub", left="01", right="sub-01")],
        None,
    )
    yield (
        "stimulus_lookup",
        "For an input outside a BIDS layout, how are the candidate stimulus files found? I'm asking for the general lookup procedure.",
        [call("explain_outside_stimulus_candidates")],
        call("explain_outside_stimulus_candidates", target="t1.c1"),
    )

    for direction in ("increase", "decrease"):
        for same_tr in (True, False):
            variation = "same positive TR throughout" if same_tr else "a TR that changes but stays positive"
            arguments = dict(
                changed_endpoint="stop",
                direction=direction,
                start_held_fixed=True,
                tr_held_fixed=same_tr,
                tr_positive=True,
            )
            negative = call("explain_hrf_interval_change_effect", **arguments, _missing="tr_held_fixed")
            yield (
                "interval",
                f"Compare two generated HRF intervals with the starting endpoint fixed. I {direction} the stop time and use {variation}. What can I infer about the number of samples?",
                [call("explain_hrf_interval_change_effect", **arguments)],
                negative,
            )


def cases(registry):
    for index, (theme, question, calls, negative) in enumerate(examples(), 1):
        for target in calls:
            registry.resolve(target)
        yield {
            "case_id": f"targeted-train-argument-{index:03}",
            "family_id": f"targeted-train-contract-{theme}",
            "group": "function_arguments",
            "history": [],
            "question": question,
            "policy": {"kind": "fresh", "calls": calls},
            "expected_targets": [],
            "rejected_calls": [negative] if negative else [],
        }
