"""Keep project temporary files and dependency caches inside the checkout."""

import json
import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
CACHE_PATHS = {
    "XDG_CACHE_HOME": "cache",
    "PIP_CACHE_DIR": "cache/pip",
    "TORCH_HOME": "cache/torch",
    "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton",
    "CUDA_CACHE_PATH": "cache/cuda",
    "HF_HOME": "cache/huggingface",
    "HF_HUB_CACHE": "cache/huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
    "HF_ASSETS_CACHE": "cache/huggingface/assets",
    "HF_XET_CACHE": "cache/huggingface/xet",
    "TRANSFORMERS_CACHE": "cache/huggingface/hub",
    "NUMBA_CACHE_DIR": "cache/numba",
    "MPLCONFIGDIR": "cache/matplotlib",
    "TMUX_TMPDIR": "tmp",
}


def private_directory(root, relative):
    current = root
    for component in Path(relative).parts:
        if component in ("", ".", "..") or Path(component).is_absolute():
            raise ValueError("Private runtime paths must stay inside the checkout")
        current = current / component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or current.is_symlink():
            raise ValueError(f"Private runtime path must be a real directory: {current}")
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError(f"Private runtime directory must be owned by you and not writable by others: {current}")
        if not current.resolve().is_relative_to(root):
            raise ValueError(f"Private runtime directory escaped the checkout: {current}")
    return current


def configure_private_runtime(root=None):
    root = ROOT if root is None else Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Expected an existing repository directory")
    scratch = private_directory(root, ".runtime")
    temporary = private_directory(root, ".runtime/tmp")
    paths = {name: str(private_directory(root, f".runtime/{relative}"))
             for name, relative in CACHE_PATHS.items()}
    paths.update({name: str(temporary) for name in ("TMPDIR", "TMP", "TEMP")})
    paths["PYTHONPYCACHEPREFIX"] = str(private_directory(root, ".runtime/cache/pycache"))
    paths["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.update(paths)
    tempfile.tempdir = str(temporary)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = paths["PYTHONPYCACHEPREFIX"]
    return {"root": str(scratch), "temporary": str(temporary), "cache": str(scratch / "cache")}


def python_tool(source):
    """Invoke a package command with this interpreter, even from another working directory."""
    path = Path(source)
    if path.parts[0] not in {"data_tools", "eval_tools", "diagnostics"} or path.suffix != ".py" or len(path.parts) != 2:
        raise ValueError(f"Unknown package command: {source}")
    module = ".".join(path.with_suffix("").parts)
    return [sys.executable, "-Bu", str(ROOT / "private_runtime.py"), "--module", module]


REPORT_MODULES = {name + ".py": "diagnostics." + name for name in
                  ("inspect_student_errors", "inspect_qwen_cases", "inspect_qwen_failures")}


TEST_ALIASES = {
    "test_inspect_qwen_cases": "test_archived_qwen_cases",
    "test_inspect_qwen_failures": "test_archived_qwen_failures",
    "test_inspect_student_errors": "test_archived_student_report",
    "transfer_dev_test": "test_transfer_dev",
    "transfer_train_test": "test_transfer_train",
    "transfer_train_replay_test": "test_transfer_train_replay",
}


def legacy_test_name(argument):
    name, separator, member = argument.partition(".")
    module = TEST_ALIASES.get(name, name)
    if (module.startswith("test_") and module.isidentifier()
            and (ROOT / "tests" / (module + ".py")).is_file()):
        suffix = separator + member if member and member != "py" else ""
        return "tests." + module + suffix
    return argument


def private_command(arguments, locations):
    arguments = list(arguments)
    if Path(arguments[0]).name in ("python", "python3"):
        # Keep the original report commands after moving their implementations.
        index = 1
        while index < len(arguments) and arguments[index] in ("-B", "-u", "-Bu"):
            index += 1
        if index < len(arguments):
            script = arguments[index]
            if script in REPORT_MODULES:
                arguments[index:index+1] = ["-m", REPORT_MODULES[script]]
            elif script.endswith(".py") and legacy_test_name(script) != script:
                arguments[index:index+1] = ["-m", "unittest", legacy_test_name(script)]
            elif arguments[index:index+2] == ["-m", "unittest"] and "discover" not in arguments[index+2:]:
                arguments[index+2:] = [legacy_test_name(arg) for arg in arguments[index+2:]]
    if Path(arguments[0]).name == "tmux":
        if any(arg.startswith(("-S", "-L")) for arg in arguments[1:]):
            raise ValueError("The launcher fixes the tmux socket inside this checkout")
        socket = Path(locations["root"]) / "tmux"
        if len(os.fsencode(socket)) >= 104:
            raise ValueError("The checkout path is too long for a private tmux socket")
        if socket.is_symlink() or (socket.exists() and not stat.S_ISSOCK(socket.lstat().st_mode)):
            raise ValueError("The private tmux socket path must not be a symlink or regular file")
        arguments[1:1] = ["-S", str(socket)]
    return arguments


if __name__ == "__main__":
    try:
        locations = configure_private_runtime()
        if len(sys.argv) == 1:
            print(json.dumps(locations, indent=2))
        elif sys.argv[1] == "--module":
            import runpy
            module = sys.argv[2]
            if module.split(".", 1)[0] not in {"data_tools", "eval_tools", "diagnostics"}:
                raise ValueError(f"Unknown command module: {module}")
            sys.argv = [module, *sys.argv[3:]]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            command = private_command(sys.argv[1:], locations)
            os.execvpe(command[0], command, os.environ)
    except (OSError, ValueError) as error:
        sys.exit(f"Private runtime error: {error}")
