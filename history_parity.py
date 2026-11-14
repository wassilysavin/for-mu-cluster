"""Bounded offline parity with the pinned official Gemma3 implementation."""

from importlib.metadata import version
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import time, time_ns

from ledger import file_record, publish_json

REFERENCE_VERSION = "4.57.3"
REFERENCE_POLICIES = {
    "auto": {"name": "auto", "version": 1, "backend": "automatic", "chunk_tokens": None},
    "math-chunked-v1": {"name": "math-chunked-v1", "version": 1, "backend": "MATH",
                        "chunk_tokens": 256, "cache": "transformers.DynamicCache"},
}
TOLERANCES = {
    "logits": {"rtol": 2e-4, "atol": 2e-3},
    "loss": {"rtol": 2e-4, "atol": 2e-5},
    "gradient": {"rtol": 2e-3, "atol": 2e-5},
}


def path_for(assets, key):
    value = assets[key]
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, dict):
        value = value["path"]
    return Path(value)


def tensor_diagnostics(actual, expected, kind, label, *, argmax=True):
    import torch
    a, b = actual.detach().float(), expected.detach().float()
    result = {"label": label, "actual_shape": list(a.shape), "expected_shape": list(b.shape),
              **TOLERANCES[kind]}
    if a.shape != b.shape or a.numel() == 0:
        return {**result, "within_tolerance": False, "reason": "shape mismatch or empty tensor"}
    difference = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    close = torch.isclose(a, b, **TOLERANCES[kind])
    index = int(difference.nan_to_num(nan=float("inf")).reshape(-1).argmax())
    absolute = float(difference.reshape(-1)[index])
    expected_at_max = float(b.reshape(-1)[index])
    relative = absolute / abs(expected_at_max) if expected_at_max else None

    def number(value):
        return value if value is not None and math.isfinite(value) else None

    result.update(within_tolerance=bool(close.all()), elements=a.numel(),
                  mismatch_count=int((~close).sum()), nonfinite_count=int((~finite).sum()),
                  maximum_absolute_error=number(absolute),
                  rms_error=number(float(difference.square().mean().sqrt())),
                  relative_error_at_maximum_absolute=number(relative),
                  actual_at_maximum_absolute=number(float(a.reshape(-1)[index])),
                  expected_at_maximum_absolute=number(expected_at_max), flat_index_at_maximum_absolute=index)
    if argmax and kind == "logits" and a.ndim >= 2:
        actual_ids, expected_ids = a.argmax(-1), b.argmax(-1)
        result.update(argmax_agreement=bool((actual_ids == expected_ids).all()),
                      actual_argmax=actual_ids.reshape(-1).tolist(),
                      expected_argmax=expected_ids.reshape(-1).tolist())
    return result


class TensorMismatch(AssertionError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(json.dumps(diagnostics, sort_keys=True))


def compare_tensor(actual, expected, kind, label):
    import torch
    diagnostics = tensor_diagnostics(actual, expected, kind, label)
    try:
        torch.testing.assert_close(actual.detach().float(), expected.detach().float(), **TOLERANCES[kind])
    except AssertionError as error:
        raise TensorMismatch(diagnostics) from error
    return diagnostics


def decoder_hidden_states(output):
    hidden = output[0] if isinstance(output, tuple) and output else output
    if not callable(getattr(hidden, "detach", None)) or not hasattr(hidden, "shape"):
        raise TypeError("Reference decoder must return hidden states, directly or as tuple element zero")
    return hidden


def decoder_layer_record(module, inputs, output):
    import torch
    if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].ndim != 3:
        raise TypeError("Reference decoder input must be a rank-three hidden-state tensor")
    hidden = decoder_hidden_states(output)
    if not isinstance(hidden, torch.Tensor) or hidden.shape != inputs[0].shape:
        raise TypeError("Reference decoder must return hidden states, directly or as tuple element zero")
    contract = {"module": type(module).__name__, "output_container": type(output).__name__,
                "tuple_length": len(output) if isinstance(output, tuple) else None,
                "hidden_shape": list(hidden.shape)}
    return inputs[0].detach(), hidden.detach(), contract


def require_complete_native_call(token_ids):
    if not token_ids or token_ids[-1] != 50:
        raise AssertionError("Complete native generation did not reach the function-response boundary")


def official_model(runtime):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import Gemma3TextConfig
    from transformers.modeling_utils import no_init_weights
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
    import model

    class AdapterLinear(nn.Module):
        def __init__(self, path):
            super().__init__()
            self.weight = nn.Parameter(runtime.base[path], requires_grad=False)
            self.a = nn.Parameter(runtime.parameters[f"{path}.lora_A"].detach().clone())
            self.b = nn.Parameter(runtime.parameters[f"{path}.lora_B"].detach().clone())

        def forward(self, x):
            return F.linear(x, self.weight) + model.LORA_SCALE * F.linear(F.linear(x, self.a), self.b)

    config = Gemma3TextConfig(**json.loads(runtime._metadata["base_model_config"]))
    config._attn_implementation = "sdpa"
    with no_init_weights():
        reference = Gemma3TextModel(config)
    reference.embed_tokens.weight = nn.Parameter(runtime.base[model.EMBEDDING], requires_grad=False)
    # Preserve the effective BF16 scalar of the imported affine4 model.
    reference.embed_tokens.embed_scale = torch.tensor(config.hidden_size ** 0.5, dtype=torch.bfloat16).float()
    adapters = {}
    for layer in range(model.LAYERS):
        for projection in model.PROJECTIONS:
            path = f"model.layers.{layer}.{projection}"
            parent, name = projection.rsplit(".", 1)
            replacement = AdapterLinear(path)
            setattr(reference.layers[layer].get_submodule(parent), name, replacement)
            adapters[f"{path}.lora_A"] = replacement.a
            adapters[f"{path}.lora_B"] = replacement.b
        for name in model.NORMS:
            reference.layers[layer].get_submodule(name).weight = nn.Parameter(
                runtime.base[f"model.layers.{layer}.{name}"] - 1, requires_grad=False)
    reference.norm.weight = nn.Parameter(runtime.base["model.norm"] - 1, requires_grad=False)
    reference.to(runtime.device).eval()
    return reference, adapters


def custom_logits(runtime, tokens, *, count=1, cache=None, offset=0):
    import model
    hidden = runtime._embed(tokens)
    next_cache = []
    for layer in range(model.LAYERS):
        previous = None if cache is None else cache[layer]
        hidden, value = runtime._layer(hidden, layer, offset, previous, cache=True)
        next_cache.append(value)
    return runtime._logits(hidden[:, -count:]), next_cache


def reference_policy_record(name):
    if name not in REFERENCE_POLICIES:
        raise ValueError(f"Unknown reference execution policy: {name}")
    return dict(REFERENCE_POLICIES[name])


def reference_hidden_states(runtime, reference, tokens, *, count=1, cache=None, use_cache=False,
                            reference_policy="auto", check_time=None):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.cache_utils import DynamicCache

    policy = reference_policy_record(reference_policy)
    if not tokens or type(count) is not int or count <= 0:
        raise ValueError("Reference execution requires tokens and a positive output count")
    if check_time is not None:
        check_time()
    ids = torch.tensor([tokens], device=runtime.device)
    if reference_policy == "auto":
        output = reference(input_ids=ids, past_key_values=cache, use_cache=use_cache)
        return output.last_hidden_state[:, -count:], output.past_key_values
    if reference.training or getattr(reference.config, "use_bidirectional_attention", False):
        raise ValueError("Chunked reference execution requires an evaluated causal Gemma3 model")
    if cache is not None and (not use_cache or not isinstance(cache, DynamicCache)):
        raise TypeError("Chunked reference continuation requires the official DynamicCache and use_cache=True")
    chunk_tokens = policy["chunk_tokens"]
    internal_cache = use_cache or len(tokens) > chunk_tokens
    start = 0 if cache is None else cache.get_seq_length()
    tail = None
    for first in range(0, len(tokens), chunk_tokens):
        if check_time is not None:
            check_time()
        with sdpa_kernel(SDPBackend.MATH):
            output = reference(input_ids=ids[:, first:first + chunk_tokens],
                               past_key_values=cache, use_cache=internal_cache)
        hidden = output.last_hidden_state[:, -count:]
        tail = hidden if tail is None else torch.cat((tail, hidden), dim=1)[:, -count:]
        if internal_cache:
            cache = output.past_key_values
            if not isinstance(cache, DynamicCache):
                raise TypeError("The pinned reference did not return the official DynamicCache")
            if cache.get_seq_length() != start + min(first + chunk_tokens, len(tokens)):
                raise AssertionError("Official reference cache did not retain the complete token position")
    return tail, cache if use_cache else None


def reference_logits(runtime, reference, tokens, *, count=1, cache=None, use_cache=False,
                     reference_policy="auto", check_time=None):
    from torch.nn import functional as F
    import model
    hidden, cache = reference_hidden_states(runtime, reference, tokens, count=count, cache=cache,
                                            use_cache=use_cache, reference_policy=reference_policy,
                                            check_time=check_time)
    return F.linear(hidden, runtime.base[model.EMBEDDING]), cache


def validate_chunked_reference(runtime, reference, tokens, checks, check_time):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import model

    if len(tokens) < 1028:
        raise ValueError("Chunked-reference controls require at least 1028 frozen probe tokens")
    weight = runtime.base[model.EMBEDDING]

    def check_logits(actual, expected, label):
        check_time()
        record = compare_tensor(F.linear(actual, weight), F.linear(expected, weight), "logits", label)
        checks.append(record)
        if not record["argmax_agreement"]:
            raise AssertionError(f"Chunked and dense MATH greedy tokens differ: {label}")

    with torch.no_grad():
        for length in (17, 511, 512, 513, 1025):
            check_time()
            with sdpa_kernel(SDPBackend.MATH):
                dense, dense_cache = reference_hidden_states(runtime, reference, tokens[:length],
                                                              count=length, use_cache=True, check_time=check_time)
            chunked, chunked_cache = reference_hidden_states(
                runtime, reference, tokens[:length], count=length, use_cache=True,
                reference_policy="math-chunked-v1", check_time=check_time)
            for first in range(0, length, 128):
                check_logits(chunked[:, first:first + 128], dense[:, first:first + 128],
                             f"reference-chunked-dense-{length}-positions-{first}:{min(first + 128, length)}")
            following = tokens[length:length + 3]
            with sdpa_kernel(SDPBackend.MATH):
                dense_next, _ = reference_hidden_states(runtime, reference, following, count=3,
                                                        cache=dense_cache, use_cache=True, check_time=check_time)
                plain_next, _ = reference_hidden_states(runtime, reference, tokens[:length + 3],
                                                        count=3, check_time=check_time)
            chunked_next, _ = reference_hidden_states(
                runtime, reference, following, count=3, cache=chunked_cache, use_cache=True,
                reference_policy="math-chunked-v1", check_time=check_time)
            check_logits(chunked_next, dense_next, f"reference-chunked-dense-cache-offset-{length}")
            check_logits(dense_next, plain_next, f"reference-dense-cache-uncached-offset-{length}")
            print(f"Reference policy control {length}: all positions and three continuation tokens agree", flush=True)


def selected_probes(path):
    lengths = []
    with path.open() as stream:
        for offset, line in enumerate(stream):
            row = json.loads(line)
            lengths.append((row["prompt_tokens"], row["case_id"], offset))
    lengths.sort()
    chosen = {lengths[len(lengths) // 2][2], lengths[-1][2]}
    with path.open() as stream:
        return [json.loads(line) for offset, line in enumerate(stream) if offset in chosen]


def template_check(tokenizer, probes, path):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    native = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(Path(path).read_text())
    envelope = re.compile(r"<bos><start_of_turn>developer\n(.*?)<end_of_turn>\n"
                          r"<start_of_turn>user\n(.*?)<end_of_turn>\n<start_of_turn>model\n", re.DOTALL)
    checks = []
    for row in probes[:2]:
        ids = row["input_ids"][:row["prompt_tokens"]]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        matched = envelope.fullmatch(decoded)
        if matched is None:
            raise ValueError(f"The frozen prompt has an unexpected native envelope: {row['case_id']}")
        messages = [{"role": "developer", "content": matched.group(1)},
                    {"role": "user", "content": matched.group(2)}]
        rendered = native.render(messages=messages, tools=[], bos_token="<bos>", add_generation_prompt=True)
        if rendered != decoded or tokenizer.encode(rendered, add_special_tokens=False).ids != ids:
            raise AssertionError(f"The pinned native template differs from the complete prompt: {row['case_id']}")
        checks.append({"label": f"native-envelope:{row['case_id']}", "exact_render": True,
                       "exact_token_ids": True, "prompt_tokens": len(ids)})
    if len(checks) != 2:
        raise ValueError("Native envelope parity requires two complete TRAIN prompts")
    return checks


def diagnose(assets, checkpoint, directory, deadline, budget):
    import platform
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.masking_utils import create_sliding_window_causal_mask
    import model

    if version("transformers") != REFERENCE_VERSION:
        raise RuntimeError(f"Diagnostics require transformers=={REFERENCE_VERSION}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started, runtime, reference = time(), None, None
    stop = min(deadline, started + 600)
    report_path = directory / f"diagnostic-{time_ns()}.json"
    report = {"schema_version": 1, "status": "DIAGNOSTIC_INCOMPLETE", "report_path": str(report_path),
              "diagnostic_only": True, "training_performed": False, "tolerances": TOLERANCES,
              "binding": {"checkpoint": file_record(Path(checkpoint)),
                          "base": file_record(path_for(assets, "base")),
                          "probes": file_record(path_for(assets, "history_replay")),
                          "implementation": file_record(Path(__file__)),
                          "runtime": file_record(Path(model.__file__))},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "cuda": torch.version.cuda, "transformers": version("transformers")},
              "attention_policy": {"custom_local": "MATH, 256-query blocks with cropped KV",
                                   "custom_global": "EFFICIENT_ATTENTION",
                                   "reference_default": "SDPA automatic dispatch; actual kernel not instrumented",
                                   "reference_control": "MATH; diagnostic only"},
              "boundaries": [], "layers": [], "cached": []}

    def check_time():
        budget.check()
        if time() >= stop:
            raise TimeoutError("Parity diagnostics exceeded the 10-minute or remaining experiment budget")

    def mask_diagnostic(length):
        positions = torch.arange(length, device=runtime.device)
        official = create_sliding_window_causal_mask(
            reference.config, torch.empty((1, length, model.WIDTH), device=runtime.device),
            None, positions, None)
        skipped = official is None
        if skipped:
            official = positions[None, :] <= positions[:, None]
        else:
            if official.dtype != torch.bool:
                raise TypeError("The pinned SDPA mask is unexpectedly non-boolean")
            official = official[0, 0]
        custom = torch.zeros((length, length), dtype=torch.bool, device=runtime.device)
        for first in range(0, length, 256):
            last = min(first + 256, length)
            begin = max(0, first - model.WINDOW + 1)
            qp, kp = positions[first:last, None], positions[None, begin:last]
            custom[first:last, begin:last] = (kp <= qp) & (kp > qp - model.WINDOW)
        mismatches = int((custom != official).sum())
        return {"exact_equal": mismatches == 0, "mismatch_count": mismatches,
                "reference_uses_causal_shortcut": skipped,
                "last_query_key_count": int(custom[-1].sum())}

    try:
        check_time()
        runtime = model.Runtime(path_for(assets, "base"), Path(checkpoint), expected_batch_size=4)
        reference, _ = official_model(runtime)
        report["environment"].update(device=str(runtime.device),
                                     device_name=torch.cuda.get_device_name(runtime.device),
                                     capability=list(torch.cuda.get_device_capability(runtime.device)),
                                     flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
                                     efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                     math_sdp_enabled=torch.backends.cuda.math_sdp_enabled())
        report["window"] = {"custom": model.WINDOW, "reference": reference.config.sliding_window,
                            "custom_layer_types": ["sliding_attention" if model.is_local(i) else "full_attention"
                                                   for i in range(model.LAYERS)],
                            "reference_layer_types": reference.config.layer_types}
        first = selected_probes(path_for(assets, "history_replay"))[0]["input_ids"]
        if len(first) < 514:
            raise ValueError("The frozen TRAIN probe is shorter than the diagnostic boundaries")
        with torch.no_grad():
            for length in (511, 512, 513, 514):
                check_time()
                tokens = first[:length]
                actual, _ = custom_logits(runtime, tokens, count=3)
                check_time()
                expected, _ = reference_logits(runtime, reference, tokens, count=3)
                check_time()
                with sdpa_kernel(SDPBackend.MATH):
                    math_expected, _ = reference_logits(runtime, reference, tokens, count=3)
                row = {"length": length, "mask": mask_diagnostic(length),
                       "custom_vs_default": tensor_diagnostics(actual, expected, "logits", f"boundary-{length}"),
                       "custom_vs_math": tensor_diagnostics(actual, math_expected, "logits", f"math-{length}"),
                       "default_vs_math": tensor_diagnostics(expected, math_expected, "logits", f"backend-{length}")}
                report["boundaries"].append(row)
                delta = row["custom_vs_default"]
                print(f"Parity diagnostic {length}: mismatches={delta['mismatch_count']} "
                      f"max_abs={delta['maximum_absolute_error']} argmax_equal={delta['argmax_agreement']} "
                      f"mask_equal={row['mask']['exact_equal']} "
                      f"custom_math_max={row['custom_vs_math']['maximum_absolute_error']} "
                      f"default_math_max={row['default_vs_math']['maximum_absolute_error']}", flush=True)
            check_time()
            report["rope"] = []
            dimensions = torch.arange(0, model.HEAD_DIM, 2, device=runtime.device, dtype=torch.float32)
            for local, rotary in ((True, reference.rotary_emb_local), (False, reference.rotary_emb)):
                custom_frequency = (10000.0 if local else 1000000.0) ** (-dimensions / model.HEAD_DIM)
                report["rope"].append(tensor_diagnostics(custom_frequency, rotary.inv_freq, "logits",
                                                        "local-frequencies" if local else "global-frequencies",
                                                        argmax=False))
            captured = []

            def capture_layer(module, inputs, output):
                captured.append(decoder_layer_record(module, inputs, output))

            handles = [layer.register_forward_hook(capture_layer) for layer in reference.layers]
            try:
                reference_logits(runtime, reference, first[:513])
            finally:
                for handle in handles:
                    handle.remove()
            if len(captured) != model.LAYERS:
                raise ValueError("Reference layer trace is incomplete")
            report["decoder_output_contracts"] = [value[2] for value in captured]
            hidden = runtime._embed(first[:513])
            for index, (reference_input, reference_output, _) in enumerate(captured):
                check_time()
                hidden = runtime._layer(hidden, index)
                isolated = runtime._layer(reference_input, index)
                report["layers"].append({"layer": index, "local": model.is_local(index),
                                          "propagated": tensor_diagnostics(hidden, reference_output, "logits",
                                                                           f"layer-{index}", argmax=False),
                                          "same_input": tensor_diagnostics(isolated, reference_output, "logits",
                                                                           f"isolated-layer-{index}", argmax=False)})
            del captured, hidden, isolated
            custom_cache = reference_cache = None
            current, offset = first[:511], 0
            for length in (511, 512, 513, 514):
                check_time()
                actual, custom_cache = custom_logits(runtime, current, cache=custom_cache, offset=offset)
                expected, reference_cache = reference_logits(runtime, reference, current,
                                                             cache=reference_cache, use_cache=True)
                plain, _ = custom_logits(runtime, first[:length])
                report["cached"].append({"length": length,
                                         "custom_vs_reference": tensor_diagnostics(actual, expected, "logits", f"cache-{length}"),
                                         "custom_vs_uncached": tensor_diagnostics(actual, plain, "logits", f"cache-plain-{length}")})
                offset = length
                current = first[length:length + 1]
        report["status"] = "DIAGNOSTIC_COMPLETE"
    except Exception as error:
        report.update(error=type(error).__name__, message=str(error))
    finally:
        if runtime is not None:
            runtime.close()
        del runtime, reference
        torch.cuda.empty_cache()
        report["seconds"] = time() - started
        publish_json(report_path, report)
    for kind in ("propagated", "same_input"):
        first_bad = next((row for row in report["layers"] if not row[kind]["within_tolerance"]), None)
        measured = len(report["layers"])
        if first_bad is None:
            result = f"none among {measured}/{model.LAYERS} measured layers" if measured else "not measured"
        else:
            attention = "sliding" if first_bad["local"] else "global"
            result = (f"{first_bad['layer']} ({attention}); max_abs={first_bad[kind]['maximum_absolute_error']} "
                      f"mismatches={first_bad[kind]['mismatch_count']}")
        print(f"First divergent layer ({kind}): {result}", flush=True)
    for row in report.get("rope", []):
        print(f"RoPE {row['label']}: max_abs={row['maximum_absolute_error']}", flush=True)
    for row in report["cached"]:
        delta, plain = row["custom_vs_reference"], row["custom_vs_uncached"]
        print(f"Cache {row['length']}: reference_max={delta['maximum_absolute_error']} "
              f"uncached_max={plain['maximum_absolute_error']} argmax_equal={delta['argmax_agreement']}", flush=True)
    print(f"Parity diagnosis: {report['status']}; report={report_path}", flush=True)
    if report.get("error"):
        print(f"{report['error']}: {report['message']}", flush=True)
    return report


def verify(assets, checkpoint, directory, deadline, budget, *, reference_policy="auto"):
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer
    import model

    if version("transformers") != REFERENCE_VERSION:
        raise RuntimeError(f"Parity requires transformers=={REFERENCE_VERSION}")
    policy = reference_policy_record(reference_policy)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "parity.json"
    source = file_record(Path(checkpoint))
    binding = {"checkpoint": source, "base": file_record(path_for(assets, "base")),
               "tokenizer": file_record(path_for(assets, "tokenizer")),
               "probes": file_record(path_for(assets, "history_replay")),
               "native_template": file_record(path_for(assets, "history_template")),
               "implementation": file_record(Path(__file__)),
               "runtime": file_record(Path(model.__file__)), "reference_version": REFERENCE_VERSION,
               "reference_policy": policy, "tolerances": TOLERANCES}
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("binding") != binding or old.get("passed") is not True:
            raise ValueError("Parity receipt changed or failed; create an explicitly revised experiment")
        return old
    for failure_path in sorted(directory.glob("parity-failed*.json")):
        failure = json.loads(failure_path.read_text())
        if failure.get("binding") != binding:
            raise ValueError("Failed parity binding changed; require an explicitly revised experiment")
        if failure.get("error") not in ("OSError", "IOError"):
            raise RuntimeError("Independent parity previously failed; inspect parity-failed.json before a revised experiment")
    stop = min(deadline, time() + 2700)

    def check_time():
        budget.check()
        if time() >= stop:
            raise TimeoutError("Independent parity exceeded its 45-minute or experiment budget")

    runtime = reference = adapters = None
    checks = []
    started = time()
    try:
        check_time()
        tokenizer = Tokenizer.from_file(str(path_for(assets, "tokenizer")))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        for text, token in (("<bos>", 2), ("<start_function_response>", 50)):
            if tokenizer.encode(text, add_special_tokens=False).ids != [token]:
                raise ValueError(f"Native tokenizer token mismatch: {text}")
        probes = selected_probes(path_for(assets, "history_replay"))
        checks.extend(template_check(tokenizer, probes, path_for(assets, "history_template")))
        runtime = model.Runtime(path_for(assets, "base"), Path(checkpoint), expected_batch_size=4)
        reference, adapters = official_model(runtime)
        first = probes[0]["input_ids"]

        def official_logits(tokens, **kwargs):
            return reference_logits(runtime, reference, tokens, reference_policy=reference_policy,
                                    check_time=check_time, **kwargs)

        if reference_policy == "math-chunked-v1":
            validate_chunked_reference(runtime, reference, first, checks, check_time)
        with torch.no_grad():
            for length in (17, 511, 512, 513, 1025):
                check_time()
                tokens = first[:length]
                actual, _ = custom_logits(runtime, tokens, count=3)
                expected, _ = official_logits(tokens, count=3)
                checks.append(compare_tensor(actual, expected, "logits", f"boundary-{length}"))
            for row in probes:
                check_time()
                tokens = row["input_ids"][:row["prompt_tokens"]]
                actual, _ = custom_logits(runtime, tokens)
                expected, _ = official_logits(tokens)
                checks.append(compare_tensor(actual, expected, "logits", f"complete-prompt:{row['case_id']}"))
                if actual.argmax(-1).item() != expected.argmax(-1).item():
                    raise AssertionError("Reference and runtime disagree on complete-prompt greedy token")
            tokens = first[:513]
            custom_cache = ref_cache = None
            custom_generated, reference_generated = [], []
            current, offset = tokens, 0
            for index in range(8):
                check_time()
                actual, custom_cache = custom_logits(runtime, current, cache=custom_cache, offset=offset)
                expected, ref_cache = official_logits(current, cache=ref_cache, use_cache=True)
                uncached, _ = official_logits(tokens + custom_generated)
                plain, _ = custom_logits(runtime, tokens + custom_generated)
                checks.append(compare_tensor(actual, expected, "logits", f"cached-reference-{index}"))
                checks.append(compare_tensor(actual, plain, "logits", f"custom-cache-{index}"))
                checks.append(compare_tensor(expected, uncached, "logits", f"reference-cache-{index}"))
                a, b = actual.argmax(-1).item(), expected.argmax(-1).item()
                if a != b:
                    raise AssertionError("Cached greedy sequences differ")
                custom_generated.append(a)
                reference_generated.append(b)
                offset += len(current)
                current = [a]
        with torch.no_grad():
            tokens = probes[0]["input_ids"][:probes[0]["prompt_tokens"]]
            custom_cache = ref_cache = None
            current, offset, complete_call = tokens, 0, []
            for index in range(512):
                check_time()
                actual, custom_cache = custom_logits(runtime, current, cache=custom_cache, offset=offset)
                expected, ref_cache = official_logits(current, cache=ref_cache, use_cache=True)
                a, b = actual.argmax(-1).item(), expected.argmax(-1).item()
                if a != b:
                    raise AssertionError(f"Complete native generation differs at token {index}")
                complete_call.append(a)
                if a == 50:
                    break
                offset += len(current)
                current = [a]
        require_complete_native_call(complete_call)
        check_time()
        prompt, completion = first[:17], first[17:20]
        actual_loss = runtime._loss_sum(prompt, completion) / len(completion)
        actual_loss.backward()
        expected_logits, _ = official_logits(prompt + completion[:-1], count=len(completion))
        expected_loss = F.cross_entropy(expected_logits[0], torch.tensor(completion, device=runtime.device))
        expected_loss.backward()
        checks.append(compare_tensor(actual_loss, expected_loss, "loss", "supervised-target-loss"))
        for name, parameter in runtime.parameters.items():
            checks.append(compare_tensor(parameter.grad, adapters[name].grad, "gradient", name))
            parameter.grad = None
            adapters[name].grad = None
        check_time()
        with TemporaryDirectory(dir=directory) as temporary:
            saved = Path(temporary) / "reload.safetensors"
            runtime.save_checkpoint(saved, lineage=runtime.lineage)
            _, _, step, lineage, parameters, rows, cols = model.read_checkpoint(saved, torch.device("cpu"))
            if step != runtime.step or lineage != runtime.lineage:
                raise AssertionError("Checkpoint reload cursor or lineage differs")
            for field, restored in (("parameters", parameters), ("rows", rows), ("cols", cols)):
                for name, value in getattr(runtime, field).items():
                    torch.testing.assert_close(value.cpu(), restored[name], rtol=0, atol=0)
        result = {"schema_version": 1, "passed": True, "binding": binding, "checks": checks,
                  "seconds": time() - started, "checkpoint_reload_exact": True,
                  "cached_greedy_tokens": custom_generated,
                  "complete_prompt_generated_tokens": complete_call,
                  "complete_native_batch": complete_call[-1] == 50,
                  "effective_embedding_scale": 25.25, "reference_downloaded_model_weights": False,
                  "note": "Independent official operations; same dequantized tensors, BF16-rounded scale and separate LoRA parameters"}
        publish_json(receipt_path, result)
        return result
    except Exception as error:
        failure = directory / "parity-failed.json"
        if failure.exists():
            failure = directory / f"parity-failed-{time_ns()}.json"
        publish_json(failure, {"passed": False, "binding": binding, "completed_checks": checks,
                               "error": type(error).__name__, "message": str(error),
                               "mismatch": getattr(error, "diagnostics", None), "seconds": time() - started})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        del reference, adapters, runtime
        torch.cuda.empty_cache()
