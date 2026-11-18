"""Move frozen replay prompts onto the transfer registry, preserving their supervision."""

from functools import lru_cache
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
CONTRACT = '\n\n{"format":"gemprf-shared-tool-contract",'
USER_TURN = "<end_of_turn>\n<start_of_turn>user\n"
MODEL_TURN = "<end_of_turn>\n<start_of_turn>model\n"


@lru_cache(maxsize=64)
def upgrade_packed(text):
    result = subprocess.run(["node", str(ROOT / "transfer_contract.mjs"), "--upgrade-packed"],
                            input=text, capture_output=True, text=True, check=True, cwd=ROOT)
    packed = result.stdout.rstrip("\n")
    parsed = json.loads(packed)
    if parsed.get("format") != "gemprf-shared-tool-contract" or parsed.get("version") != 1:
        raise ValueError("Transfer registry returned an unexpected prompt contract")
    return packed.replace("<", "\\u003c")


def upgrade_replay(row, tokenizer):
    case = row["case_id"]
    ids, split = row["input_ids"], row["prompt_tokens"]
    prompt = tokenizer.decode(ids[:split], skip_special_tokens=False)
    if (not prompt.startswith("<bos><start_of_turn>developer\n") or not prompt.endswith(MODEL_TURN)
            or prompt.count(CONTRACT) != 1 or prompt.count(USER_TURN) != 1):
        raise ValueError(f"Invalid frozen developer/user prompt boundaries in {case}")
    prefix, rest = prompt.split(CONTRACT)
    old, user = rest.split(USER_TURN)
    old = CONTRACT[2:] + old
    target_ids = ids[split:]
    target = tokenizer.decode(target_ids, skip_special_tokens=False)
    if tokenizer.encode(prompt + target, add_special_tokens=False).ids != ids:
        raise ValueError(f"Frozen replay cannot round-trip under the pinned tokenizer: {case}")
    new = upgrade_packed(old)
    if new == old:
        raise ValueError(f"Frozen TRAIN is missing the required contract revision: {case}")
    updated = prefix + "\n\n" + new + USER_TURN + user
    prompt_ids = tokenizer.encode(updated, add_special_tokens=False).ids
    full = tokenizer.encode(updated + target, add_special_tokens=False).ids
    if (full != prompt_ids + target_ids or len(prompt_ids) + 512 > 32768
            or not 0 < len(target_ids) <= 512 or target_ids[-1] != 50):
        raise ValueError(f"Transfer contract changed supervision or exceeded the context limit: {case}")
    return {"prompt_tokens": len(prompt_ids), "input_ids": full}
