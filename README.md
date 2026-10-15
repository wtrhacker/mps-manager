# MPS Manager

Run independent Python scripts together on one NVIDIA GPU. Give each queue a
title, submit as many jobs as you need, and manage them from your terminal.
Each job can use a different script, arguments, environment, and working directory.

## Install

Requires Linux, Python 3.10+, and an NVIDIA GPU/driver with working MPS client
termination support (`nvidia-smi` and `nvidia-cuda-mps-control` installed).
The prebuilt wheel supports Linux x86_64 with glibc 2.34+:

```bash
python -m pip install https://github.com/wtrhacker/mps-manager/releases/download/v0.2.2/mps_manager-0.2.2-py3-none-linux_x86_64.whl
```

Installation provides both the `mps_runner` Python package and the `mps-manager`
command in your active Python environment.

To install from source instead, with Rust 1.97+ available:

```bash
python -m pip install git+https://github.com/wtrhacker/mps-manager.git
```

## Start a queue

Write your usual simulation scripts, then submit them from any Python file:

```python
from mps_runner import run_mps_scripts

queue = run_mps_scripts(
    [
        {"script_path": "simulate.py", "args": ["--stiffness", "0.5"], "title": "Soft"},
        {"script_path": "another.py", "args": ["--seed", "7"], "title": "Another run"},
    ],
    title="My simulations",
    gpu=0,
    max_concurrent=2,
    detach=True,
)
```

`args` is optional. Jobs use the submitting program's Python interpreter,
environment, and working directory by default. The selected GPU appears inside
each script as CUDA device **0**. MPS startup or attachment failures raise an error.

`detach=True` returns immediately. The default waits for the submitted work.
The parent queue stays available for inspection and further submissions.
`max_concurrent` limits this parent's live children; suspended children retain their
slots. The complete effective arguments are saved for every child.

## Share a GPU across parents

Create another parent with the same `gpu` value. Sharing is automatic:

```python
other = run_mps_scripts(
    [{"script_path": "extra.py", "args": ["--seed", "9"]}],
    title="Another queue",
    gpu=0,
    detach=True,
)
```

Both parents keep their own titles, PIDs, jobs, and controls in the TUI. The
shared service lives independently of either parent. Terminating one parent
cancels only its children; the service stays available for the other parents.
After the final parent closes, the owned service closes once its clients are gone.

Admission is **FIFO within each parent and round-robin between eligible parents**.
When both have work and room under their limits, the next slots go A1, B1, A2, B2.
Paused parents and held jobs are skipped. Already-running jobs continue, and
completion order depends on the scripts. GPU execution can be concurrent.

`gpu_max_concurrent` sets the **total live-job limit across parents**. Set it on
the first parent, for example `gpu_max_concurrent=8`; subsequent parents inherit
it when omitted. The initial default is 64. An explicitly conflicting limit is
an error. This is a job-count limit; choose a value that fits your jobs' GPU memory
and compute requirements. Each parent's `max_concurrent` also continues to apply.

Use the same project directory or `state_dir` to browse both parents together.
GPU sharing also works between parents whose project folders differ.

## Add more jobs

Another Python program can find the same queue by its unique title or parent PID:

```python
from mps_runner import connect_mps_manager

queue = connect_mps_manager(title="My simulations")
job = queue.submit("extra.py", args=["--seed", "8"], title="Extra run")
```

Run on the same host and in the same project directory. Queue state and logs
live in the hidden `.mps-manager/` folder; `state_dir` can select another folder.

Ordinary Python loops are enough for a parameter sweep:

```python
for stiffness in [0.5, 1.5, 4.0]:
    queue.submit("simulate.py", args=["--stiffness", str(stiffness)])
```

These jobs join the existing parent queue. Its concurrency settings decide how
many run together on the GPU.

## Manage queues in the terminal

```bash
mps-manager tui
```

The first screen shows **parent queues**, with titles and PIDs. Select a parent
to view its jobs, actual child PIDs, status, details, and stdout/stderr logs.
`mps-manager tui --pid PID` opens a particular parent directly.

| Key | Action |
| --- | --- |
| Up / Down | Select a parent or job. |
| Enter / Right | Open the queue, job details, then logs. |
| Left / Escape | Go back. |
| Tab in logs | Switch stdout/stderr. |
| `p` / `r` | Pause / resume the selected queue on the parents screen, or the selected child inside a queue. |
| `q` / Ctrl+C | Close the viewer; jobs keep running. |
| `t` | Terminate the selected parent and permanently hide it after cleanup. |
| `:` | Type a command, then press Enter. |

Commands include `:pause`, `:resume`, `:kill`, `:hold`, `:release`, `:remove`,
`:move 0`, `:top`, `:bottom`, `:shuffle`, `:title New title`, `:pause-all`,
and `:resume-all`. Job commands act on the selected child. On the parents screen,
pause/resume controls the selected queue.

Finished parents remain visible until you press `t`. Termination and removal
preserve logs and script outputs. Cleanup errors leave the parent visible.

## Python controls

```python
job.pause()           # Hold a queued job or suspend a live child.
job.resume()
job.kill()            # Cancel with MPS-aware cleanup.
job.remove()          # Remove a queued or completed entry.
queue.move(job.id, 0)
queue.shuffle()
queue.pause()         # Suspend the queue's children and stop new admissions.
queue.resume()
queue.wait()
queue.shutdown()      # Stop the parent; keep its inspection history.
queue.terminate()     # Stop and hide the parent, like pressing t.
```

## Optional sweep helper

`run_mps_scripts` creates a parent queue, `queue.submit` adds one job, and
`queue.submit_sweep` expands parameter combinations and adds those jobs to the
same queue. It supports Cartesian combinations, numeric ranges, correlated
cases, and replicates.

For scripts that accept `--name value` arguments:

```python
jobs = queue.submit_sweep(
    "simulate.py",
    parameters={"stiffness": {"$sweep": [0.5, 1.5, 4.0]}, "density": 0.2},
    output_arg="--output",
    replicates=2,
)
```

This adds six jobs to the existing queue. `stiffness` varies; `density` stays
fixed at 0.2 and is defined only once. Both values are recorded for each job.

Parameter names become flags exactly as written: `density` becomes `--density`,
and `mean-depth-kt` becomes `--mean-depth-kt`. Values can be strings or numbers.
`output_arg="--output"` adds each job's distinct output folder under that flag;
omit it if your script does not accept an output argument. The folder is also
available to each script through the `MPS_RUN_DIR` environment variable.

For other argument formats, an optional `make_args` function can supply the
argument list. Existing custom functions continue to work.

## Fixed arguments

Parent `common_args` are inherited by ordinary jobs and sweeps. Per submission
or sweep, `common_args=[...]` replaces that prefix, and `common_args=[]` opts out.
Job-specific arguments are appended in the order you supply; flags are not parsed
or rewritten by the manager. Choose one place for each fixed argument: the common
prefix or the sweep's `parameters`.

```python
queue.set_common_args(["--density", "0.3"])  # Future jobs only.
job = queue.submit("different.py", args=["--seed", "10"], common_args=[])
```

Already-submitted jobs keep their original argument snapshot. Job details show
both the common prefix and full command arguments. Sweeps additionally record
the complete resolved parameter object, including fixed values.

## Runtime behavior

Scripts own their outputs, random seeds, and checkpoint/restart behavior. Failed jobs are never
retried automatically, and ordinary script failures leave other jobs running.

External STOP/CONT signals are reflected in the viewer. Suspension retains GPU
memory and contexts; already submitted GPU work can continue. Use the manager's
kill command for MPS-aware cancellation.

To use a compatible externally managed MPS service, supply `mps_pipe_directory`
when creating a queue. Its parent queues share admission too, and leave the
external service running. Ordinary same-GPU sharing needs no pipe argument.

Existing v0.1 parent queues keep their original behavior; close them before
starting the new automatic shared-service mode. Their saved history remains readable.
