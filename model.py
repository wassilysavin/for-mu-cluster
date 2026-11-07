"""FunctionGemma 270M LoRA runtime for a single Volta GPU."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


LAYERS, WIDTH, HEADS, HEAD_DIM, VOCAB = 18, 640, 4, 256, 262144
WINDOW, CONTEXT, GROUP, RANK = 512, 32768, 64, 16
LORA_SCALE, LOSS_SCALE, LEARNING_RATE = 2.0, 128.0, 1e-4
SOURCE_FORMAT = "gemprf-mlx-checkpoint-handoff-v1"
RUN_FORMAT = "gemprf-torch-volta-checkpoint-v1"
OPTIMIZER_POLICY = {
    "optimizer": "functional-fp32-factored-adafactor-v1",
    "peak_learning_rate": LEARNING_RATE, "warmup_updates": 54,
    "eps": [1e-30, 0.001], "clip_threshold": 1.0, "decay_rate": -0.8,
    "beta1": None, "weight_decay": 0.0, "scale_parameter": False,
    "relative_step": False, "warmup_init": False, "global_gradient_clip": None,
    "gradient_contract": "sum of per-micro gradients of loss_sum/group_total_supervised_tokens",
}
PROJECTIONS = {
    "self_attn.q_proj": (1024, 640), "self_attn.k_proj": (256, 640),
    "self_attn.v_proj": (256, 640), "self_attn.o_proj": (640, 1024),
    "mlp.gate_proj": (2048, 640), "mlp.up_proj": (2048, 640),
    "mlp.down_proj": (640, 2048),
}
NORMS = {
    "input_layernorm": WIDTH, "post_attention_layernorm": WIDTH,
    "pre_feedforward_layernorm": WIDTH, "post_feedforward_layernorm": WIDTH,
    "self_attn.q_norm": HEAD_DIM, "self_attn.k_norm": HEAD_DIM,
}
EMBEDDING = "model.embed_tokens"
ROLES = {"parameter": "lora", "row": "adafactor.row", "col": "adafactor.col"}


def layer_path(layer: int) -> str:
    return f"model.layers.{layer}"


def is_local(layer: int) -> bool:
    return layer % 6 != 5


def adapter_shapes() -> dict[str, tuple[int, int]]:
    return {
        f"{layer_path(layer)}.{name}.lora_{side}": shape
        for layer in range(LAYERS)
        for name, (out_size, in_size) in PROJECTIONS.items()
        for side, shape in (("A", (RANK, in_size)), ("B", (out_size, RANK)))
    }


def tensor_digest(value: torch.Tensor) -> str:
    return sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def decode_affine(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint32 or scales.dtype != torch.bfloat16 or biases.dtype != torch.bfloat16:
        raise ValueError("Expected affine4 uint32 words and bfloat16 scales and biases")
    shifts = torch.arange(0, 32, 4, device=packed.device)
    words = packed.view(torch.int32).to(torch.int64).unsqueeze(-1)
    values = ((words >> shifts) & 15).reshape(packed.shape[0], -1, GROUP).float()
    return (values * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).flatten(1)


def adafactor_update(parameter, gradient, row, col, step):
    beta = 1.0 - step ** -0.8
    square = gradient.square() + 1e-30
    row = row * beta + square.mean(-1) * (1.0 - beta)
    col = col * beta + square.mean(-2) * (1.0 - beta)
    direction = (torch.rsqrt(row / row.mean())[:, None] * torch.rsqrt(col)[None, :]) * gradient
    direction = direction / (direction.square().sum().sqrt() / parameter.numel() ** 0.5).clamp(min=1)
    return parameter - direction * (LEARNING_RATE * min(step / 54.0, 1.0)), row, col


def load_base(path: Path, manifest: dict, device: torch.device) -> dict[str, torch.Tensor]:
    artifact = manifest["file"]
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    quantization = manifest["quantization"]
    if (path.stat().st_size != artifact["bytes"] or digest != artifact["sha256"]
            or (quantization["bits"], quantization["group_size"], quantization["mode"]) != (4, GROUP, "affine")):
        raise ValueError("Frozen affine base differs from the checkpoint")
    matrices = {EMBEDDING: (VOCAB, WIDTH)}
    norms = {"model.norm": WIDTH}
    for layer in range(LAYERS):
        matrices.update({f"{layer_path(layer)}.{key}": shape for key, shape in PROJECTIONS.items()})
        norms.update({f"{layer_path(layer)}.{key}": size for key, size in NORMS.items()})
    keys = {f"{name}.{suffix}" for name in matrices for suffix in ("weight", "scales", "biases")}
    keys.update(f"{name}.weight" for name in norms)
    base = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        if set(source.keys()) != keys:
            raise ValueError("Frozen base tensor names differ")
        for name, (rows, cols) in matrices.items():
            packed, scales, biases = [source.get_slice(f"{name}.{suffix}") for suffix in ("weight", "scales", "biases")]
            if (packed.get_shape() != [rows, cols // 8]
                    or scales.get_shape() != [rows, cols // GROUP]
                    or biases.get_shape() != scales.get_shape()):
                raise ValueError(f"Wrong frozen matrix shape: {name}")
            dense = torch.empty((rows, cols), dtype=torch.float16)
            for start in range(0, rows, 2048):
                part = slice(start, min(start + 2048, rows))
                dense[part] = decode_affine(packed[part], scales[part], biases[part])
            base[name] = dense.to(device)
        for name, width in norms.items():
            weight = source.get_tensor(f"{name}.weight")
            if weight.shape != (width,) or weight.dtype != torch.bfloat16:
                raise ValueError(f"Wrong frozen norm: {name}")
            base[name] = weight.to(device=device, dtype=torch.float32) + 1.0
    return base


def checkpoint_origin(metadata: dict[str, str]) -> dict:
    raw = metadata["checkpoint_manifest"]
    if sha256(raw.encode()).hexdigest() != metadata["checkpoint_manifest_sha256"]:
        raise ValueError("Original checkpoint manifest hash differs")
    origin = json.loads(raw)
    if (origin["step"] != 4847 or origin["complete"] is not True
            or origin["cursor"]["accumulated_microsteps"] != 0):
        raise ValueError("Expected the complete step 4847 source checkpoint")
    if origin["hyperparameters"] != OPTIMIZER_POLICY:
        raise ValueError("Checkpoint Adafactor policy differs")
    config = json.loads(metadata["base_model_config"])
    expected = {
        "model_type": "gemma3_text", "num_hidden_layers": LAYERS, "hidden_size": WIDTH,
        "intermediate_size": 2048, "num_attention_heads": HEADS, "num_key_value_heads": 1,
        "head_dim": HEAD_DIM, "vocab_size": VOCAB, "max_position_embeddings": CONTEXT,
        "sliding_window": WINDOW, "query_pre_attn_scalar": 256, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0,
        "hidden_activation": "gelu_pytorch_tanh", "attention_dropout": 0.0,
        "attention_bias": False, "use_bidirectional_attention": False,
        "rope_scaling": None, "attn_logit_softcapping": None, "final_logit_softcapping": None,
        "layer_types": ["sliding_attention" if is_local(i) else "full_attention" for i in range(LAYERS)],
    }
    if config.get("tie_word_embeddings", True) is not True or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint model configuration differs")
    return origin


def read_checkpoint(path: Path, device: torch.device):
    shapes = adapter_shapes()
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        kind = metadata.get("format")
        if kind not in (SOURCE_FORMAT, RUN_FORMAT):
            raise ValueError("Unknown checkpoint format")
        origin = checkpoint_origin(metadata)
        step = int(metadata["step"])
        if step < origin["step"] or (kind == SOURCE_FORMAT and step != origin["step"]):
            raise ValueError("Invalid checkpoint step")
        lineage = json.loads(metadata["lineage"]) if kind == RUN_FORMAT else {}
        if not isinstance(lineage, dict) or any(not isinstance(value, str) for value in lineage.values()):
            raise ValueError("Invalid checkpoint lineage")
        digests = (json.loads(metadata["tensor_sha256"]) if kind == RUN_FORMAT else {
            f"{ROLES[row['role']]}.{row['name']}": row["sha256"] for row in origin["payloads"]
        })
        keys = {f"{prefix}.{name}" for prefix in ROLES.values() for name in shapes}
        if set(source.keys()) != keys or set(digests) != keys:
            raise ValueError("Checkpoint tensor names differ")
        parameters, rows, cols = {}, {}, {}
        for name, shape in shapes.items():
            for prefix, expected_shape, table in (
                ("lora", shape, parameters), ("adafactor.row", shape[:1], rows),
                ("adafactor.col", shape[1:], cols),
            ):
                key = f"{prefix}.{name}"
                value = source.get_tensor(key)
                if (value.dtype != torch.float32 or value.shape != expected_shape
                        or tensor_digest(value) != digests[key] or not torch.isfinite(value).all()
                        or (prefix != "lora" and not (value > 0).all())):
                    raise ValueError(f"Invalid checkpoint tensor: {key}")
                table[name] = value.to(device).requires_grad_(prefix == "lora")
    return metadata, origin["step"], step, lineage, parameters, rows, cols


class Runtime:
    def __init__(self, base_path: Path, checkpoint_path: Path, *, device: str = "cuda"):
        self.device = torch.device(device)
        if (sys.platform != "linux" or self.device.type != "cuda" or not torch.cuda.is_available()
                or torch.__version__.split("+")[0] != "2.7.1"
                or torch.cuda.get_device_capability(self.device) != (7, 0)
                or "sm_70" not in torch.cuda.get_arch_list() or torch.version.cuda != "11.8"):
            raise RuntimeError("This runtime requires a Linux V100 and PyTorch 2.7.1 CUDA 11.8 with sm_70")
        (self._metadata, self.initial_step, self.step, self.lineage,
         self.parameters, self.rows, self.cols) = read_checkpoint(Path(checkpoint_path), self.device)
        self.tokenizer_sha256 = self._metadata["tokenizer_sha256"]
        self.base = load_base(Path(base_path), json.loads(self._metadata["base_model_manifest"]), self.device)

    def _linear(self, value, path):
        frozen = F.linear(value, self.base[path]).float()
        adjusted = F.linear(F.linear(value.float(), self.parameters[f"{path}.lora_A"]),
                            self.parameters[f"{path}.lora_B"])
        return (frozen + LORA_SCALE * adjusted).half()

    def _norm(self, value, path):
        value = value.float()
        return (value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * self.base[path]).half()

    def _rope(self, value, offset, local):
        positions = torch.arange(offset, offset + value.shape[2], device=self.device, dtype=torch.float32)
        dimensions = torch.arange(0, HEAD_DIM, 2, device=self.device, dtype=torch.float32)
        frequencies = (10000.0 if local else 1000000.0) ** (-dimensions / HEAD_DIM)
        phase = torch.outer(positions, frequencies).repeat(1, 2)
        left, right = value.chunk(2, dim=-1)
        turned = torch.cat((-right, left), dim=-1)
        return value * phase.cos().half() + turned * phase.sin().half()

    def _qkv(self, value, stem, offset, local):
        def project(key, heads):
            projected = self._linear(value, f"{stem}.{key}_proj")
            return projected.view(1, value.shape[1], heads, HEAD_DIM).transpose(1, 2)

        query = self._norm(project("q", HEADS), f"{stem}.q_norm")
        key = self._norm(project("k", 1), f"{stem}.k_norm")
        return self._rope(query, offset, local), self._rope(key, offset, local), project("v", 1)

    def _attention(self, query, key, value, offset, local):
        key = key.expand(-1, HEADS, -1, -1).contiguous()
        value = value.expand(-1, HEADS, -1, -1).contiguous()
        query_count, key_count = query.shape[2], key.shape[2]
        if not local:
            if key_count != offset + query_count or (offset and query_count != 1):
                raise ValueError("Unexpected global attention extent")
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(query.contiguous(), key, value,
                                                      is_causal=offset == 0, scale=1 / 16)
        origin = offset + query_count - key_count
        outputs = []
        for first in range(0, query_count, 256):
            last = min(first + 256, query_count)
            begin = max(0, offset + first - WINDOW + 1 - origin)
            end = offset + last - origin
            qp = torch.arange(offset + first, offset + last, device=self.device)[:, None]
            kp = torch.arange(origin + begin, origin + end, device=self.device)[None, :]
            mask = (kp <= qp) & (kp > qp - WINDOW)
            with sdpa_kernel(SDPBackend.MATH):
                block = F.scaled_dot_product_attention(query[:, :, first:last], key[:, :, begin:end],
                                                       value[:, :, begin:end], attn_mask=mask, scale=1 / 16)
            outputs.append(block)
        return torch.cat(outputs, dim=2)

    def _layer(self, value, layer, offset=0, past=None, *, cache=False):
        stem = layer_path(layer)
        local = is_local(layer)
        inputs = self._norm(value, f"{stem}.input_layernorm")
        query, key, content = self._qkv(inputs, f"{stem}.self_attn", offset, local)
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            content = torch.cat((past[1], content), dim=2)
        attended = self._attention(query, key, content, offset, local)
        projected = self._linear(attended.transpose(1, 2).reshape(1, value.shape[1], -1), f"{stem}.self_attn.o_proj")
        value = value + self._norm(projected, f"{stem}.post_attention_layernorm")
        mlp = self._norm(value, f"{stem}.pre_feedforward_layernorm")
        gate = F.gelu(self._linear(mlp, f"{stem}.mlp.gate_proj").float(), approximate="tanh").half()
        up = self._linear(mlp, f"{stem}.mlp.up_proj")
        down = self._linear(gate * up, f"{stem}.mlp.down_proj")
        value = value + self._norm(down, f"{stem}.post_feedforward_layernorm")
        if not cache:
            return value
        if local:
            key, content = key[:, :, -WINDOW:].contiguous(), content[:, :, -WINDOW:].contiguous()
        return value, (key, content)

    def _embed(self, tokens: list[int]):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return F.embedding(ids, self.base[EMBEDDING]) * 25.25

    def _logits(self, hidden):
        return F.linear(self._norm(hidden, "model.norm"), self.base[EMBEDDING]).float()

    def _loss_sum(self, prompt: list[int], completion: list[int]):
        hidden = self._embed(prompt + completion[:-1])
        for layer in range(LAYERS):
            hidden = checkpoint(self._layer, hidden, layer, use_reentrant=False, preserve_rng_state=False)
        hidden = hidden[:, -len(completion):]
        targets = torch.tensor(completion, device=self.device, dtype=torch.long)
        return F.cross_entropy(self._logits(hidden)[0], targets, reduction="sum")

    @staticmethod
    def _validate_tokens(tokens, maximum):
        if (not tokens or len(tokens) > maximum
                or any(type(token) is not int or not 0 <= token < VOCAB for token in tokens)):
            raise ValueError("Invalid complete token sequence")

    def train_step(self, examples: Sequence[tuple[list[int], list[int]]]) -> float:
        if len(examples) != 2:
            raise ValueError("One update requires two microbatches")
        for prompt, completion in examples:
            self._validate_tokens(prompt, CONTEXT)
            self._validate_tokens(completion, CONTEXT - len(prompt))
        count = sum(len(completion) for _, completion in examples)
        losses = []
        try:
            for prompt, completion in examples:
                loss = self._loss_sum(prompt, completion) / count
                losses.append(loss.detach())
                (loss * LOSS_SCALE).backward()
            mean_loss = torch.stack(losses).sum().item()
            if not torch.stack([torch.isfinite(loss) for loss in losses]).all().item():
                raise FloatingPointError("Nonfinite target loss")
            self._update()
            return mean_loss
        finally:
            for parameter in self.parameters.values():
                parameter.grad = None

    @torch.no_grad()
    def _update(self):
        next_step = self.step + 1
        updated = {}
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing adapter gradient: {name}")
            updated[name] = adafactor_update(parameter, parameter.grad / LOSS_SCALE,
                                            self.rows[name], self.cols[name], next_step)
        valid = [torch.isfinite(parameter).all() & (row > 0).all() & torch.isfinite(row).all()
                 & (col > 0).all() & torch.isfinite(col).all() for parameter, row, col in updated.values()]
        if not torch.stack(valid).all().item():
            raise FloatingPointError("Invalid Adafactor update")
        for name, (parameter, row, col) in updated.items():
            self.parameters[name].copy_(parameter)
            self.rows[name], self.cols[name] = row, col
        self.step = next_step

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], *, max_new_tokens: int, stop_ids: Collection[int]) -> list[int]:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("A positive generation budget is required")
        self._validate_tokens(prompt_ids, CONTEXT - max_new_tokens)
        generated, cached = [], None
        current, offset = prompt_ids, 0
        for _ in range(max_new_tokens):
            hidden = self._embed(current)
            next_cache = []
            for layer in range(LAYERS):
                previous = None if cached is None else cached[layer]
                hidden, state = self._layer(hidden, layer, offset, previous, cache=True)
                next_cache.append(state)
            cached = next_cache
            logits = self._logits(hidden[:, -1:])
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite generation logits")
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token in stop_ids:
                break
            offset += len(current)
            current = [token]
        return generated

    def save_checkpoint(self, path: Path, *, lineage: Mapping[str, str]) -> None:
        path = Path(path)
        if self.lineage and dict(lineage) != self.lineage:
            raise ValueError("Checkpoint lineage differs from the resumed run")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in lineage.items()):
            raise ValueError("Checkpoint lineage requires string keys and values")
        tensors = {
            f"{prefix}.{name}": value.detach().cpu().contiguous()
            for prefix, values in (("lora", self.parameters), ("adafactor.row", self.rows), ("adafactor.col", self.cols))
            for name, value in values.items()
        }
        metadata = dict(self._metadata, format=RUN_FORMAT, step=str(self.step),
                        lineage=json.dumps(dict(lineage), sort_keys=True),
                        tensor_sha256=json.dumps({key: tensor_digest(value) for key, value in tensors.items()}, sort_keys=True))
        with TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory) / path.name
            save_file(tensors, staged, metadata=metadata)
            with staged.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(staged, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.lineage = dict(lineage)
