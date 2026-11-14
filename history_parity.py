"""Bounded offline parity with the pinned official Gemma3 implementation."""

from importlib.metadata import version
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import time, time_ns

from ledger import file_record, publish_json

REFERENCE_VERSION = "4.57.3"
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


def compare_tensor(actual, expected, kind, label):
    import torch
    a, b = actual.detach().float(), expected.detach().float()
    torch.testing.assert_close(a, b, **TOLERANCES[kind], msg=label)
    return {"label": label, "maximum_absolute_error": float((a - b).abs().max()),
            "rms_error": float((a - b).square().mean().sqrt()), **TOLERANCES[kind]}


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


def reference_logits(runtime, reference, tokens, *, count=1, cache=None, use_cache=False):
    import torch
    from torch.nn import functional as F
    import model
    output = reference(input_ids=torch.tensor([tokens], device=runtime.device),
                       past_key_values=cache, use_cache=use_cache)
    return F.linear(output.last_hidden_state[:, -count:], runtime.base[model.EMBEDDING]), output.past_key_values


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


def verify(assets, checkpoint, directory, deadline, budget):
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer
    import model

    if version("transformers") != REFERENCE_VERSION:
        raise RuntimeError(f"Parity requires transformers=={REFERENCE_VERSION}")
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
               "tolerances": TOLERANCES}
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
        with torch.no_grad():
            for length in (17, 511, 512, 513, 1025):
                check_time()
                tokens = first[:length]
                actual, _ = custom_logits(runtime, tokens, count=3)
                expected, _ = reference_logits(runtime, reference, tokens, count=3)
                checks.append(compare_tensor(actual, expected, "logits", f"boundary-{length}"))
            for row in probes:
                check_time()
                tokens = row["input_ids"][:row["prompt_tokens"]]
                actual, _ = custom_logits(runtime, tokens)
                expected, _ = reference_logits(runtime, reference, tokens)
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
                expected, ref_cache = reference_logits(runtime, reference, current, cache=ref_cache, use_cache=True)
                uncached, _ = reference_logits(runtime, reference, tokens + custom_generated)
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
                expected, ref_cache = reference_logits(runtime, reference, current, cache=ref_cache, use_cache=True)
                a, b = actual.argmax(-1).item(), expected.argmax(-1).item()
                if a != b:
                    raise AssertionError(f"Complete native generation differs at token {index}")
                complete_call.append(a)
                if a == 50:
                    break
                offset += len(current)
                current = [a]
        check_time()
        prompt, completion = first[:17], first[17:20]
        actual_loss = runtime._loss_sum(prompt, completion) / len(completion)
        actual_loss.backward()
        expected_logits, _ = reference_logits(runtime, reference, prompt + completion[:-1], count=len(completion))
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
                               "error": type(error).__name__, "message": str(error), "seconds": time() - started})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        del reference, adapters, runtime
        torch.cuda.empty_cache()
