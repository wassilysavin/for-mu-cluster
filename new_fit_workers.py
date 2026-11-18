"""Generate resumable TRAIN responses on four independent V100 replicas."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext, suppress
from hashlib import sha256
from pathlib import Path
from threading import Lock, Thread
import json
import os
import re
import shutil
import stat
import subprocess
import sys

from private_runtime import private_directory
from student_evidence import digest, verify

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUTPUT = RUNS / "new-train-fit-v1"
GPUS = 4
LIMIT = 512
STOP = 50


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class FitBudget:
    def __init__(self, output):
        import diagnose
        if Path(output) != OUTPUT:
            raise ValueError("Unexpected TRAIN fit output directory")
        self.output = Path(output)
        self.lock = Lock()
        previous = RUNS / "train-fit-reference-v1"
        if not (previous / "disk-budget.json").is_file():
            previous = RUNS / "semantic-transfer-v1"
        self.previous = diagnose.DiskBudget(previous)

    def require(self, size):
        import diagnose
        from history import contained
        patterns = ("transfer*", "test_transfer*", "assets/transfer*", "*new*fit*.py", "*new*fit*.mjs",
                    "inspect_transfer.py", "saved_transfer.py", "test_inspect_transfer.py", "test_saved_transfer.py")
        sources = {p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()}
        paths = [p for p in RUNS.glob("semantic-transfer-v1*") if p != self.previous.output]
        if self.output.exists():
            paths.append(self.output)
        allocation = {}
        for top in [*paths, *sources]:
            for path in [top, *top.rglob("*")] if top.is_dir() else [top]:
                contained(path, ROOT)
                entry = diagnose.allocation(path)
                allocation[entry["device"], entry["inode"]] = entry["bytes"]
        git_copy = sum(p.stat().st_size for p in sources)
        used = self.previous.usage()["charged_bytes"] + sum(allocation.values()) + git_copy
        available = min(diagnose.DISK_LIMIT - used, shutil.disk_usage(ROOT).free - diagnose.RESERVE)
        if size < 0 or size + 16384 > available:
            raise RuntimeError(f"TRAIN fit requires {size + 16384:,} bytes; {available:,} available within quota/reserve")
        return available

    @contextmanager
    def writing(self, size):
        with self.lock:
            self.require(size)
            yield


def validate_ids(ids):
    if (not isinstance(ids, list) or not 0 < len(ids) <= LIMIT
            or any(type(x) is not int or not 0 <= x < 262144 for x in ids)
            or STOP in ids[:-1] or (ids[-1] != STOP and len(ids) != LIMIT)):
        raise ValueError("Invalid native generation length, token, or stopping boundary")


def generation(ids, n, tokenizer):
    validate_ids(ids)
    complete = ids[-1] == STOP
    return {"raw_ids": ids, "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
            "complete": complete, "finish_reason": "complete_batch" if complete else "length",
            "provider_kind": "functiongemma", "input_tokens": n, "max_new_tokens": LIMIT}


def make_record(binding, cid, prompt_hash, ids):
    validate_ids(ids)
    record = {"binding": binding, "case_id": cid, "prompt_ids_sha256": prompt_hash, "raw_ids": ids}
    return {**record, "sha256": digest(record)}


def torn_prefix(fragment, binding, cid, prompt_hash):
    head = encode({"binding": binding, "case_id": cid, "prompt_ids_sha256": prompt_hash, "raw_ids": []})[:-2]
    if len(fragment) <= len(head):
        return head.startswith(fragment)
    if not fragment.startswith(head):
        return False
    tail = fragment[len(head):]
    if b"]" in tail:
        # All generated IDs survived. A partial checksum must also match these IDs.
        try:
            ids = json.loads(b"[" + tail.split(b"]", 1)[0] + b"]")
            return encode(make_record(binding, cid, prompt_hash, ids)).startswith(fragment)
        except (TypeError, ValueError):
            return False
    pieces = tail.split(b",")
    if len(pieces) > LIMIT:
        return False
    return all(re.fullmatch(rb"(0|[1-9][0-9]{0,5})", part) and int(part) < 262144
               and (i == len(pieces)-1 or int(part) != STOP)
               for i, part in enumerate(pieces) if part or i < len(pieces)-1)


def journal_prefix(path, cases, binding, prompt_hashes, budget):
    if path.is_symlink():
        raise ValueError(f"Symlink in generation path: {path}")
    if not path.exists():
        return {}
    saved, repair = {}, None
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Unexpected generation journal: {path}")
        for i, raw in enumerate(stream):
            if i >= len(cases):
                raise ValueError("Extra generations after scheduled shard")
            cid = cases[i]
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError):
                if raw.endswith(b"\n") or not torn_prefix(raw, binding, cid, prompt_hashes[cid]):
                    raise ValueError(f"Corrupt generation for {cid}") from None
                repair = ("truncate", info.st_size - len(raw))
                break
            expected = make_record(binding, cid, prompt_hashes[cid], row["raw_ids"])
            if row != expected:
                raise ValueError(f"Generation binding or checksum differs for {cid}")
            saved[cid] = row["raw_ids"]
            if not raw.endswith(b"\n"):
                repair = ("newline", info.st_size)
    if repair:
        with budget.writing(1), os.fdopen(os.open(path, os.O_WRONLY | os.O_NOFOLLOW), "wb") as stream:
            current = os.fstat(stream.fileno())
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_nlink) != (
                    info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, 1):
                raise ValueError(f"Generation journal changed during verification: {path}")
            action, offset = repair
            stream.seek(offset)
            if action == "newline":
                stream.write(b"\n")
            else:
                stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
    return saved


def append_record(path, record, budget):
    line = encode(record) + b"\n"
    with budget.writing(len(line)):
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_APPEND | os.O_CREAT, 0o600), "ab") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Unexpected generation journal: {path}")
            new_file = info.st_size == 0
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if new_file:
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def child(gpu, lease, liveness):
    import run
    run.verify_worker_lease(RUNS, lease)
    def parent_watch():
        os.read(liveness, 1)
        os._exit(72)
    Thread(target=parent_watch, daemon=True).start()
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    import torch
    from model import Runtime
    run.configure_parallel_numerics()
    if not 0 <= gpu < GPUS or torch.cuda.device_count() < GPUS:
        raise ValueError("Four visible V100 GPUs are required")
    torch.cuda.set_device(gpu)
    spec = json.loads(input())
    checkpoint = verify(ROOT, ROOT / spec["checkpoint"]["path"], spec["checkpoint"])
    base = verify(ROOT, ROOT / spec["base"]["path"], spec["base"])
    runtime = Runtime(base, checkpoint, device=f"cuda:{gpu}", expected_batch_size=4)
    try:
        if runtime.step != 8151 or runtime.tokenizer_sha256 != spec["tokenizer"]["sha256"]:
            raise ValueError("TRAIN fit needs the evaluated step-8151 checkpoint and its original tokenizer")
        uuid = str(getattr(torch.cuda.get_device_properties(gpu), "uuid", "unavailable"))
        print(json.dumps({"gpu": gpu, "uuid": uuid}), flush=True)
        for line in sys.stdin:
            cid, ids = json.loads(line)
            raw = runtime.generate(ids, max_new_tokens=LIMIT, stop_ids={STOP})
            validate_ids(raw)
            print(json.dumps([cid, raw], separators=(",", ":")), flush=True)
    finally:
        runtime.close()


def exchange(process, message):
    process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"GPU fit worker exited without a reply (exit {process.poll()})")
    return json.loads(response)


@contextmanager
def replicas(gpus, spec, lease):
    import run
    run.verify_worker_lease(RUNS, lease)
    read_fd, write_fd = os.pipe()
    processes, uuids = {}, set()
    try:
        for gpu in gpus:
            process = subprocess.Popen([sys.executable, "-Bu", __file__, str(gpu), str(lease), str(read_fd)],
                                       pass_fds=(lease, read_fd), cwd=ROOT, text=True, encoding="utf-8",
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            processes[gpu] = process
            ready = exchange(process, spec)
            if ready["gpu"] != gpu or (ready["uuid"] != "unavailable" and ready["uuid"] in uuids):
                raise ValueError("Worker did not bind to the assigned physical GPU")
            uuids.add(ready["uuid"])
        yield processes
        for process in processes.values():
            process.stdin.close()
        for process in processes.values():
            if process.wait() != 0:
                raise RuntimeError("GPU worker failed before clean shutdown")
    finally:
        os.close(write_fd)
        os.close(read_fd)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                with suppress(BrokenPipeError):
                    process.stdin.close()
            process.stdout.close()


def generate_arm(output, arm, binding, checkpoint, base, tokenizer, prompts, lease, budget):
    from tokenizers import Tokenizer
    if output != budget.output or not re.fullmatch(r"(ordinary|contrast)-(1729|2718)", arm):
        raise ValueError("Unexpected fit arm or output root")
    if not re.fullmatch(r"[0-9a-f]{64}", binding):
        raise ValueError("Invalid fit manifest digest")
    if len(prompts) != 640:
        raise ValueError("Expected the entire 640-presentation TRAIN schedule")
    hashes = {cid: sha256(encode(list(ids))).hexdigest() for cid, ids in prompts.items()}
    if any(not 0 < len(ids) <= 32768-LIMIT for ids in prompts.values()):
        raise ValueError("Invalid prepared TRAIN prompt length")
    decoder = Tokenizer.from_file(str(verify(ROOT, ROOT / tokenizer["path"], tokenizer)))
    decoder.no_padding()
    decoder.no_truncation()
    spec = {"checkpoint": checkpoint, "base": base, "tokenizer": tokenizer}
    shards = [list(prompts)[gpu::GPUS] for gpu in range(GPUS)]
    bindings = [digest({"manifest": binding, "arm": arm, "gpu": gpu, **spec}) for gpu in range(GPUS)]
    target = output / arm
    with (nullcontext() if target.exists() else budget.writing(8192)):
        directory = private_directory(ROOT, target.relative_to(ROOT))
    files = [directory / f"gpu-{gpu}.jsonl" for gpu in range(GPUS)]
    stored = [journal_prefix(path, cases, identity, hashes, budget)
              for path, cases, identity in zip(files, shards, bindings, strict=True)]
    pending = [gpu for gpu, cases in enumerate(shards) if len(stored[gpu]) != len(cases)]

    def run_shard(gpu, process):
        cases = shards[gpu]
        for i, cid in enumerate(cases[len(stored[gpu]):], start=len(stored[gpu])+1):
            received, raw = exchange(process, [cid, list(prompts[cid])])
            if received != cid:
                raise ValueError(f"Unexpected GPU response for {cid}")
            record = make_record(bindings[gpu], cid, hashes[cid], raw)
            append_record(files[gpu], record, budget)
            stored[gpu][cid] = raw
            if i == 1 or i % 25 == 0 or i == len(cases):
                print(f"[{arm} GPU {gpu}] TRAIN {i}/{len(cases)}", flush=True)

    if pending:
        # On an exception, shut the GPU processes down before awaiting blocked reader threads.
        with ThreadPoolExecutor(max_workers=GPUS) as pool, replicas(pending, spec, lease) as processes:
            futures = [pool.submit(run_shard, gpu, process) for gpu, process in processes.items()]
            for future in as_completed(futures):
                future.result()
    return {cid: generation(stored[gpu][cid], len(prompts[cid]), decoder)
            for gpu, cases in enumerate(shards) for cid in cases}


if __name__ == "__main__":
    child(*map(int, sys.argv[1:]))
