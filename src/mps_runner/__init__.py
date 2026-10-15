"""Run independent Python scripts in titled queues on one NVIDIA GPU.

The client is standard-library-only. The native mps-manager owns queues, MPS, and the terminal viewer. It reuses
Scientific Workflow's process, sweep-expansion, and terminal components.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping
import uuid

__all__ = ["run_mps_scripts", "connect_mps_manager", "list_mps_managers", "expand_parameters", "Manager", "Job", "MpsManagerError"]


class MpsManagerError(RuntimeError):
    """A manager, MPS startup, or explicit control operation failed."""


def _binary() -> Path:
    configured = os.environ.get("MPS_MANAGER_BINARY")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise MpsManagerError(f"MPS_MANAGER_BINARY does not exist: {candidate}")
    bundled = Path(__file__).parent / "_bin" / "mps-manager-native"
    if bundled.is_file():
        return bundled
    repository = Path(__file__).resolve().parents[2] / "native"
    for mode in ("release", "debug"):
        candidate = repository / "target" / mode / "mps-manager-native"
        if candidate.is_file():
            return candidate
    found = shutil.which("mps-manager-native")
    if found:
        return Path(found)
    raise MpsManagerError("native manager is missing; install a built wheel or build native/Cargo.toml")


def _start_identity(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return None if fields[0] in {"Z", "X"} else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise MpsManagerError(f"cannot read manager state at {path}: {error}") from error


def list_mps_managers(*, state_dir=".mps-manager") -> list[dict]:
    """List titled parent queues; discovery never starts or stops anything."""
    parents = []
    for path in Path(state_dir).expanduser().resolve().glob("queues/*/state.json"):
        try:
            state = _read_state(path)
        except MpsManagerError:
            continue
        if state.get("dismissed"):
            continue
        state["alive"] = state["host"] == socket.gethostname() and _start_identity(state["pid"]) == state["process_start"]
        parents.append(state)
    return sorted(parents, key=lambda parent: (parent["alive"], parent["submitted_at"]), reverse=True)


class Job:
    """A stable logical job ID, with its actual child PID available after launch."""

    def __init__(self, manager: Manager, job_id: str):
        self.manager, self.id = manager, job_id

    def status(self) -> dict:
        try:
            return self.manager._request("job", job_id=self.id)
        except MpsManagerError:
            state = _read_state(self.manager._path)
            if state["state"] == "closed":
                for job in state["jobs"]:
                    if job["id"] == self.id:
                        return job
            raise

    @property
    def pid(self) -> int | None:
        return self.status()["pid"]

    @property
    def run_dir(self) -> Path:
        return self.manager._root / "jobs" / self.id / "artifacts"

    def pause(self):
        self.manager.pause(self.id)

    def resume(self):
        self.manager.resume(self.id)

    def kill(self):
        self.manager.kill(self.id)

    def remove(self):
        self.manager.remove(self.id)

    def logs(self, stream="stderr") -> str:
        return self.manager.logs(self.id, stream=stream)


class Manager:
    """Connection to one persistent parent queue; disconnecting leaves it running."""

    def __init__(self, state: dict):
        self.id, self.pid = state["id"], state["pid"]
        self._path = Path(state["root"]) / "state.json"
        self._identity = state["process_start"]
        self._host = state["host"]
        self._socket = state["socket"]
        self._root = Path(state["root"])

    def _request(self, action: str, **payload):
        if self._host != socket.gethostname():
            raise MpsManagerError(f"queue runs on {self._host}; connect on that host")
        if _start_identity(self.pid) != self._identity:
            raise MpsManagerError("parent process exited or its PID was reused")
        fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"T", "t"}:
            if action == "resume" and not payload.get("job_id"):
                result = subprocess.run([str(_binary()), "wake", str(self._path)], capture_output=True, text=True)
                if result.returncode:
                    raise MpsManagerError(result.stderr.strip())
            else:
                raise MpsManagerError("parent process is suspended; resume the parent first")
        request = json.dumps({"queue_id": self.id, "action": action, **payload}).encode() + b"\n"
        if len(request) > 8 * 1024 * 1024:
            raise ValueError("submission is too large; submit it in smaller batches")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(20)
                connection.connect(self._socket)
                connection.sendall(request)
                with connection.makefile("rb") as stream:
                    response = json.loads(stream.readline(8 * 1024 * 1024 + 1))
        except (OSError, ValueError) as error:
            raise MpsManagerError(f"cannot communicate with queue '{self.id}': {error}") from error
        if not response.get("ok"):
            raise MpsManagerError(response.get("error", "manager operation failed"))
        return response["result"]

    def status(self) -> dict:
        if _start_identity(self.pid) != self._identity:
            state = _read_state(self._path)
            if state["state"] == "closed":
                return state
        return self._request("status")

    @property
    def title(self) -> str:
        return self.status()["title"]

    @property
    def common_args(self) -> list[str]:
        """Fixed argument prefix inherited by future submissions."""
        return list(self.status().get("common_args", []))

    def set_common_args(self, args):
        """Change defaults for future jobs; already submitted jobs keep their arguments."""
        if isinstance(args, (str, bytes)):
            raise TypeError("common_args must be a sequence of arguments, not a string")
        self._request("common_args", args=[str(arg) for arg in args])

    def set_title(self, title: str):
        self._request("title", title=title)

    def _spec(self, script_path, *, args=(), title=None, cwd=None, python=None, env=None, job_id=None, common_args=None, parameters=None) -> dict:
        script = Path(script_path).expanduser().resolve(strict=True)
        interpreter = Path(python or sys.executable).expanduser().absolute()
        directory = Path(cwd or Path.cwd()).expanduser().resolve(strict=True)
        if not script.is_file() or not directory.is_dir() or not interpreter.is_file():
            raise ValueError("script, working directory, or interpreter is invalid")
        if isinstance(args, (str, bytes)):
            raise TypeError("args must be a sequence of arguments, not a shell command")
        if isinstance(common_args, (str, bytes)):
            raise TypeError("common_args must be a sequence of arguments, not a string")
        owned = {"CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
                 "CUDA_MPS_PROTOCOL_VERSION", "MPS_RUN_DIR", "MPS_JOB_ID"}
        environment = {key: value for key, value in os.environ.items()
                       if key not in owned and not key.startswith("MPS_MANAGER_")}
        environment.update({str(key): str(value) for key, value in (env or {}).items()})
        return {"id": job_id or "job-" + uuid.uuid4().hex[:16], "script_path": str(script),
                "args": [str(arg) for arg in args], "title": title, "cwd": str(directory),
                "python": str(interpreter), "env": environment,
                "common_args": None if common_args is None else [str(arg) for arg in common_args],
                "parameters": parameters}

    def submit(self, script_path, *, args=(), title=None, cwd=None, python=None, env=None, common_args=None) -> Job:
        """Append one script; arguments are passed directly to Python without a shell."""
        spec = self._spec(script_path, args=args, title=title, cwd=cwd, python=python, env=env, common_args=common_args)
        ids = self._request("submit", runs=[spec])
        return Job(self, ids[0])

    def submit_many(self, runs: Iterable[Mapping]) -> list[Job]:
        specs = [self._spec(**dict(run)) for run in runs]
        return [Job(self, job_id) for job_id in self._request("submit", runs=specs)]

    def submit_sweep(self, script_path, *, parameters, make_args: Callable | None = None, output_arg=None, replicates=1, title=None, cwd=None, python=None, common_args=None) -> list[Job]:
        """Expand parameters and append the resulting jobs to this queue.

        By default, each string or number becomes --name value, using the
        parameter name exactly as written. output_arg="--output" optionally
        passes each job's private output folder under that flag.
        Custom argument formats can use make_args(case, run_dir) instead. The
        case is a dict with .replicate and .combination attributes for explicit seeds.
        common_args=None inherits the parent prefix; [] opts out, and a list
        replaces that prefix for this sweep. The final arguments and complete
        resolved parameter object, including fixed values, are saved per job.
        No scientific seed or simulation parameter is changed automatically.
        """
        if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
            raise ValueError("replicates must be a positive integer")
        if make_args is not None and not callable(make_args):
            raise TypeError("make_args must be callable or None")
        if output_arg is not None and (not isinstance(output_arg, str) or not output_arg.startswith("-")
                                       or output_arg in {"-", "--"} or any(c.isspace() for c in output_arg)):
            raise ValueError("output_arg must be an option such as '--output', or None")
        configurations = expand_parameters(parameters)
        specs = []
        for replicate in range(replicates):
            for combination, values in enumerate(configurations):
                if not isinstance(values, dict):
                    raise ValueError("sweep parameters must resolve to objects")
                case = _SweepCase(values, replicate, combination)
                args = []
                if make_args is None:
                    for name, value in case.items():
                        if not name or name.startswith("-") or any(c.isspace() for c in name):
                            raise ValueError("automatic sweep parameter names must be flag names without leading '-' or spaces")
                        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                            raise TypeError(f"automatic sweep argument '{name}' must be a string or number; use make_args for custom formats")
                        if output_arg is not None and output_arg.lstrip("-") == name:
                            raise ValueError(f"'{name}' is already in parameters; choose parameters or output_arg, not both")
                        args.extend([f"--{name}", str(value)])
                job_id = "job-" + uuid.uuid4().hex[:16]
                run_dir = self._root / "jobs" / job_id / "artifacts"
                run_dir.mkdir(parents=True, exist_ok=False)
                if make_args is not None:
                    args = make_args(case, run_dir)
                if isinstance(args, (str, bytes)):
                    raise TypeError("args must be a sequence of arguments, not a shell command")
                if output_arg is not None:
                    args = [*args, output_arg, str(run_dir)]
                label = title or f"{Path(script_path).name} {json.dumps(values, separators=(',', ':'))} replicate={replicate + 1}"
                spec = self._spec(script_path, args=args, title=label, cwd=cwd, python=python, job_id=job_id, common_args=common_args, parameters=dict(case))
                specs.append(spec)
        return [Job(self, job_id) for job_id in self._request("submit", runs=specs)]

    def pause(self, job_id=None):
        """Suspend one child, or stop admission and suspend all children."""
        self._request("pause", job_id=job_id)

    def resume(self, job_id=None):
        self._request("resume", job_id=job_id)

    def hold(self, job_id):
        self._request("hold", job_id=job_id)

    def release(self, job_id):
        self._request("release", job_id=job_id)

    def kill(self, job_id):
        self._request("kill", job_id=job_id)

    def remove(self, job_id):
        self._request("remove", job_id=job_id)

    def move(self, job_id, position):
        self._request("move", job_id=job_id, position=position)

    def reorder(self, job_ids):
        self._request("reorder", job_ids=list(job_ids))

    def shuffle(self):
        self._request("shuffle")

    def logs(self, job_id, *, stream="stderr") -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        if _start_identity(self.pid) != self._identity:
            state = _read_state(self._path)
            if state["state"] == "closed":
                if not any(job["id"] == job_id for job in state["jobs"]):
                    raise MpsManagerError("unknown job ID")
                path = self._root / "jobs" / job_id / f"{stream}.log"
                if not path.exists():
                    return ""
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 65536))
                    return "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-200:])
        return self._request("logs", job_id=job_id, stream=stream)

    def wait(self, *, timeout=None) -> list[dict]:
        """Wait for all current runnable jobs; return held/suspended jobs as-is."""
        started = time.monotonic()
        while True:
            state = self.status()
            if state["error"]:
                raise MpsManagerError(state["error"])
            runnable = any(job["state"] in {"starting", "running"} for job in state["jobs"])
            pending = bool(state["pending"]) and not state["admission_paused"]
            if not runnable and not pending:
                return state["jobs"]
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError("jobs are still running; the queue remains active")
            time.sleep(0.05)

    def shutdown(self, *, timeout=30):
        """Explicitly cancel this queue's remaining jobs and stop its owned service."""
        if _start_identity(self.pid) != self._identity:
            state = _read_state(self._path)
            if state["state"] == "closed":
                if getattr(self, "_process", None) is not None:
                    self._process.wait(timeout=5)
                return
            raise MpsManagerError("parent exited with an incomplete queue; inspect its saved state")
        self._request("shutdown")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = _read_state(self._path)
            if state["state"] == "closed":
                if getattr(self, "_process", None) is not None:
                    self._process.wait(timeout=5)
                return
            if state["state"] == "fault":
                raise MpsManagerError(state["error"] or "manager cleanup failed")
            time.sleep(0.05)
        raise TimeoutError("manager cleanup is still running")

    def terminate(self, *, timeout=30):
        """Finish this parent and persistently remove it from the browser, preserving data."""
        if _start_identity(self.pid) == self._identity:
            self._request("terminate")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                state = _read_state(self._path)
                if state["state"] == "closed" and state.get("dismissed"):
                    if getattr(self, "_process", None) is not None:
                        self._process.wait(timeout=5)
                    return
                if state["state"] == "fault":
                    raise MpsManagerError(state["error"] or "cleanup failed; parent remains visible")
                time.sleep(0.05)
            raise TimeoutError("parent cleanup is still running; it remains visible")
        state = _read_state(self._path)
        if any(job["state"] not in {"succeeded", "failed", "cancelled", "removed"} for job in state["jobs"]):
            raise MpsManagerError("offline parent has unfinished jobs; inspect its PIDs first")
        state["dismissed"] = True
        temporary = self._path.with_suffix(f".{uuid.uuid4().hex}.partial")
        temporary.write_text(json.dumps(state))
        temporary.replace(self._path)


class _SweepCase(dict):
    def __init__(self, values, replicate, combination):
        super().__init__(values)
        self.replicate, self.combination = replicate, combination


def expand_parameters(parameters) -> list:
    """Use Workflow's existing $sweep/$cases/linspace/logspace implementation."""
    process = subprocess.run([str(_binary()), "expand"], input=json.dumps(parameters), text=True, capture_output=True)
    if process.returncode:
        raise ValueError(process.stderr.strip())
    return json.loads(process.stdout)


def connect_mps_manager(pid=None, *, title=None, state_dir=".mps-manager") -> Manager:
    """Connect by parent PID or unique human title, without creating another queue."""
    matches = [parent for parent in list_mps_managers(state_dir=state_dir)
               if parent["alive"] and (pid is None or parent["pid"] == int(pid)) and (title is None or parent["title"] == title)]
    if len(matches) != 1:
        raise MpsManagerError("select one live queue by PID or a unique title; use mps-manager tui to browse")
    manager = Manager(matches[0])
    manager.status()
    return manager


def run_mps_scripts(runs=(), *, gpu=0, max_concurrent=4, title=None, state_dir=".mps-manager", detach=False,
                    python=None, mps_pipe_directory=None, mps=True, common_args=(), gpu_max_concurrent=None) -> Manager:
    """Create a parent queue and submit scripts. NVIDIA MPS is strict by default.

    detach=True returns immediately for later submissions. detach=False waits
    for this initial batch, but the manager stays available until shutdown().
    Parents on the same GPU automatically share a service, with round-robin
    admission between parents and FIFO within each queue. gpu_max_concurrent
    sets the shared service limit (default 64 for the first parent); later
    parents inherit it when omitted. common_args prefixes each child's args.
    mps=False is explicit ordinary-process mode for development and CPU tests;
    providing gpu_max_concurrent enables shared scheduling for CPU tests too.
    """
    if sys.platform != "linux":
        raise MpsManagerError("the external-script manager currently requires Linux")
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or not 1 <= max_concurrent <= 64:
        raise ValueError("max_concurrent must be an integer between 1 and 64")
    if isinstance(common_args, (str, bytes)):
        raise TypeError("common_args must be a sequence of arguments, not a string")
    if gpu_max_concurrent is not None and (isinstance(gpu_max_concurrent, bool) or not isinstance(gpu_max_concurrent, int) or not 1 <= gpu_max_concurrent <= 64):
        raise ValueError("gpu_max_concurrent must be an integer between 1 and 64")
    if not isinstance(mps, bool):
        raise TypeError("mps must be a boolean; MPS is never disabled automatically")
    root = Path(state_dir).expanduser().resolve()
    queue_id = "queue-" + uuid.uuid4().hex[:16]
    queue_dir = root / "queues" / queue_id
    queue_dir.mkdir(parents=True, mode=0o700)
    manifest = queue_dir / "launch.json"
    configuration = {"id": queue_id, "title": title or f"GPU {gpu} queue", "state_dir": str(root),
                     "worker": str(Path(__file__).parent / "_worker.py"), "python": str(Path(python or sys.executable).absolute()),
                     "gpu": str(gpu), "max_concurrent": max_concurrent, "mps": bool(mps),
                     "mps_pipe_directory": str(Path(mps_pipe_directory).resolve()) if mps_pipe_directory else None,
                     "common_args": [str(arg) for arg in common_args], "gpu_max_concurrent": gpu_max_concurrent}
    manifest.write_text(json.dumps(configuration))
    with (queue_dir / "manager.log").open("wb") as log:
        process = subprocess.Popen([str(_binary()), "daemon", str(manifest)], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    deadline = time.monotonic() + 40
    while not (queue_dir / "ready.json").exists():
        if (queue_dir / "startup-error.json").exists():
            process.wait(timeout=10)
            raise MpsManagerError(json.loads((queue_dir / "startup-error.json").read_text())["error"])
        if process.poll() is not None:
            raise MpsManagerError((queue_dir / "manager.log").read_text().strip() or "manager exited during startup")
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait(timeout=10)
            raise MpsManagerError("manager startup timed out")
        time.sleep(0.03)
    manager = Manager(_read_state(queue_dir / "state.json"))
    manager._process = process
    try:
        manager.submit_many(runs)
    except BaseException:
        manager.shutdown()
        raise
    if not detach:
        manager.wait()
    return manager
