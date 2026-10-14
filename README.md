# MPS Manager

Run independent Python scripts on one NVIDIA GPU. Keep a large editable queue,
add work from another Python program, and manage titled parent queues in a terminal.
Scripts keep ownership of their science, arguments, outputs, and checkpoints.
There are no automatic retries or analysis features in this interface.

The Python client requires Python 3.10+ and Linux. Installed wheels bundle the
native queue manager and have no Python runtime dependencies. The existing
Workflow Python reader/conversion companion is not required. MPS mode requires
NVIDIA's `nvidia-smi`, `nvidia-cuda-mps-control`, a Volta-or-newer GPU, and a
driver supporting client termination (470+). The manager uses MPS protocol 2.

## Submit scripts

Any filename can contain these calls:

```python
from mps_runner import run_mps_scripts

manager = run_mps_scripts(
    [
        {"script_path": "simulate.py", "args": ["--stiffness", str(k)],
         "title": f"stiffness={k}"}
        for k in [0.5, 1.5, 4.0, 8.0]
    ],
    title="Stiffness sweep",
    gpu=0,
    max_concurrent=4,
    detach=True,
)
```

Arguments are passed directly to the selected Python interpreter, without a
shell. Each submission can use a different script. `args` is optional. The
submitting program's interpreter, environment, and working directory are the
defaults; `python`, `env`, and `cwd` can be supplied per job. GPU/MPS affinity
and manager variables remain owned by the queue.

The selected GPU is exposed to children as local CUDA device 0. Numeric `gpu`
selection respects the caller's visible-device allocation; a GPU UUID is also
accepted. MPS initialization and verified-attachment failures are errors, never
ordinary-CUDA fallbacks. `mps=False` explicitly selects ordinary process mode
for development and CPU tests.

`detach=True` returns immediately. The default waits for the initial work and
returns the same manager handle. An idle or finished parent remains available
for new submissions and inspection until explicitly shut down or terminated.

## Add jobs later

```python
from mps_runner import connect_mps_manager

manager = connect_mps_manager(title="Stiffness sweep")
job = manager.submit("another_simulation.py", args=["--seed", "7"], title="Extra run")
```

Use a unique title or `pid=...`. The submitting program and manager must run on
the same host; project discovery defaults to `.mps-manager` in the current
directory. `state_dir` can select another project. The manager owns one queue,
and its concurrency limit counts all live children, including suspended ones.

## Open the terminal browser

```bash
mps-manager tui
```

The first page lists **parent queues only**, identified by title and PID.
Finished and closed parents remain visible. Users can inspect their children
and logs after completion. `mps-manager tui --pid PID` jumps directly to one
parent.

| Key | Action |
| --- | --- |
| Up / Down, PageUp / PageDown | Select parents or children; scroll logs. |
| Enter / Right | Open parent, job details, then logs. |
| Left / Escape | Return to the previous page. |
| Tab in logs | Switch stdout/stderr. |
| `q` or Ctrl+C | Close the viewer; queues and jobs keep running. |
| `t` | Terminate the selected **parent queue**, then permanently hide it after cleanup. |
| `:` | Begin a command; Enter submits it. |

Examples: `:pause`, `:resume`, `:kill`, `:remove`, `:move 0`, `:top`, `:bottom`,
`:shuffle`, `:title My queue`, `:name My job`, `:pause-all`, and `:resume-all`.
Job commands target the selected child; on the parents page, pause/resume
target the whole selected queue. A pending job can be held/released with
pause/resume or the explicit `hold`/`release` commands.

`t` preserves logs and scientific outputs; it is not a file-deletion command.
Removing a queued or completed job also preserves its data. Termination errors
leave the parent visible. No PID memorization is necessary.

Status, actual live PIDs, queue order, and logs are always available. This mode
does not guess scientific progress, elapsed simulation time, or ETA. Historic
PIDs are marked exited in details and are not shown as live PIDs in the table.

## Python controls

```python
job.pause()                 # Hold a pending job or SIGSTOP a live child.
job.resume()                # Release it or SIGCONT the same live child.
job.kill()                  # Explicit cancellation, with MPS-aware teardown.
job.remove()                # Remove pending/completed entry; preserve data.
manager.move(job.id, 0)      # Reorder pending work only.
manager.shuffle()
manager.pause()             # Stop admission and suspend the owned children.
manager.resume()
manager.wait(timeout=60)    # Does not cancel jobs if it times out.
manager.shutdown()          # Stop queue and service; retain browser history.
manager.terminate()         # Stop and persistently hide this parent, like `t`.
```

External SIGSTOP/SIGCONT and exits are reflected in the browser. Suspending a
process retains its GPU memory and CUDA contexts, and already submitted GPU
work can continue. Cancellation first resumes suspended driver threads, then
asks MPS to terminate only the owned group's CUDA clients before signaling
its processes. Arbitrary direct SIGKILL bypasses this teardown; prefer the
manager's cancellation command for MPS clients.

Ordinary script failures leave healthy siblings running. GPU/service faults
close admission and are reported separately. No failed script is restarted.
Changing a checkpoint/restart argument is an ordinary new submission; the
manager neither creates checkpoints nor interprets their contents.

## Optional sweeps

```python
def make_args(case, run_dir):
    return ["--stiffness", str(case["stiffness"]),
            "--density", str(case["density"]), "--output", str(run_dir)]

jobs = manager.submit_sweep(
    "simulate.py",
    parameters={"stiffness": {"$sweep": [0.5, 1.5]},
                "density": {"$sweep": [0.1, 0.2]}},
    make_args=make_args,
    replicates=3,
)
```

Workflow's original expander resolves four combinations, repeated three times
into twelve ordinary jobs with distinct output areas. It also supports `$cases`,
`linspace`, and `logspace` with the existing validation and ordering. Literal
arrays remain literal. `case.replicate` and `case.combination` are zero-based
indices available for explicitly constructing seeds or labels; model parameters
and random seeds are never changed automatically. The argument builder runs
only in the submitting Python process and is not serialized to the manager.

## Files and service ownership

Generated manifests, queue metadata, per-job environment snapshots, logs, and
default output areas live beneath `.mps-manager/queues/<queue-id>/`. Metadata
is versioned and published atomically. Control sockets and NVIDIA MPS pipes
use private host-local temporary directories, which work on network-mounted
projects too. Checkpoints and other scientific output locations are script-owned.

The manager starts a private MPS endpoint by default. To attach a compatible
existing endpoint, supply `mps_pipe_directory`, or configure
`CUDA_MPS_PIPE_DIRECTORY` before creating the manager. The child probe verifies
the GPU UUID and its actual server registration before the submitted script
starts. Attached services are not shut down. An owned service is stopped only
after its clients are gone; shared external clients prevent unsafe shutdown.
If an owned service still has another parent's clients, termination reports an
error and leaves the owner visible and controllable. Finish those clients first,
then explicitly terminate the owner again.

## Two repositories

This repository owns the human-facing Python API, NVIDIA MPS adapter, persistent
queue engine, queue terminal browser, packaging, examples, and acceptance tests.
Its native crate depends on the separate
[Scientific-Workflow fork](https://github.com/wtrhacker/Scientific-Workflow).
The fork provides its existing parameter expander, owned process groups,
terminal restoration lease, command editor, status palette, and resource sampler
through a small opt-in Rust API. Neither repository contains a copy of the other.
No study manifests, recording readers, analysis pipeline, or Workflow Python
companion are used when launching scripts.

## Install and develop

For the current unpublished source, keep the two Git checkouts side by side:

```text
scientific-workflow/   # Separate fork, including the external-tools feature.
mps-manager/          # This repository; no vendored Workflow source.
```

From `mps-manager/`:

```bash
python -m pip install .
mps-manager tui
```

Source builds require Rust 1.97+ and the separate sibling Workflow checkout.
The development dependency is an explicit path in `native/Cargo.toml`; the
tested fork revision is recorded in `WORKFLOW_DEPENDENCY.md`. When publishing
the two GitHub repositories, this dependency can be changed to that Git revision
without moving or copying either repository into the other.
A packaged Linux wheel includes the native binary and needs no Rust compiler at
runtime. Build a wheel with `python -m pip wheel . --no-deps -w dist`.
`MPS_MANAGER_BINARY` selects an existing developer build. Source clients also
look in `native/target/release` and `native/target/debug`.

```bash
cargo build --locked --manifest-path native/Cargo.toml
export MPS_MANAGER_BINARY="$PWD/native/target/debug/mps-manager-native"
export PYTHONPATH="$PWD/src"
python -m pip install '.[test]'
python -m unittest discover -s tests -p 'test_*.py' -v
cargo test --locked --manifest-path native/Cargo.toml
cargo clippy --locked --manifest-path native/Cargo.toml --all-targets -- -D warnings
```

`tests/live_gpu.py`, `tests/live_tui_gpu.py`, and `tests/live_attach_gpu.py` are explicit, bounded real-GPU
acceptance runners. Launch only on a GPU host assigned to your tests. They check
real CUDA outputs, verified MPS registration, concurrency, queue changes, signal
controls, safe cancellation, cross-process additions, failure isolation, terminal
restoration, retained history, and explicit termination. Their evidence and
terminal screen captures go under the invoking project's `.mps-manager/`.
They do not measure throughput or validate scientific models.
