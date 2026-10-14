"""Explicit real-GPU acceptance; only launched on the user-approved test GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from mps_runner import run_mps_scripts, connect_mps_manager, list_mps_managers

FIXTURE = Path(__file__).parent / "fixtures" / "cuda_job.py"
state_dir = Path.cwd() / ".mps-manager" / "validation" / ("live-" + uuid.uuid4().hex[:8])
state_dir.mkdir(parents=True)


def until(function, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = function()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("condition timed out")


manager = run_mps_scripts(gpu=0, max_concurrent=4, title="Live MPS acceptance", state_dir=state_dir, detach=True)
evidence = {"host": os.uname().nodename, "parent_pid": manager.pid, "state_dir": str(state_dir), "checks": []}
try:
    info = manager.status()
    evidence["gpu_uuid"] = info["gpu_uuid"]
    manager.pause()
    jobs = [manager.submit(FIXTURE, args=["--seconds", "9", "--value", str(60 + index), "--label", f"worker {index}"], title=f"CUDA child {index}") for index in range(4)]
    extra = manager.submit(FIXTURE, args=["--seconds", "0.4"], title="queued extra")
    removed = manager.submit(FIXTURE, title="removed before launch")
    removed.remove()
    manager.move(extra.id, 0)
    manager.shuffle()
    manager.reorder([job.id for job in jobs] + [extra.id])
    manager.resume()
    until(lambda: all(job.status()["state"] == "running" for job in jobs))
    until(lambda: all((job.run_dir / "ticks.txt").exists() for job in jobs))
    evidence["children"] = [job.pid for job in jobs]
    evidence["checks"].append("four concurrent verified MPS clients on one GPU")
    assert extra.status()["state"] == "pending"
    jobs[0].pause()
    until(lambda: jobs[0].status()["state"] == "suspended")
    time.sleep(0.15)
    frozen = (jobs[0].run_dir / "ticks.txt").read_text()
    other = (jobs[1].run_dir / "ticks.txt").read_text()
    time.sleep(0.3)
    assert (jobs[0].run_dir / "ticks.txt").read_text() == frozen
    assert (jobs[1].run_dir / "ticks.txt").read_text() != other
    assert extra.status()["state"] == "pending", "suspended clients retain their live slots"
    jobs[0].resume()
    evidence["checks"].append("individual suspension preserves siblings and retains concurrency slot")
    os.kill(jobs[1].pid, signal.SIGSTOP)
    until(lambda: jobs[1].status()["state"] == "suspended")
    os.kill(jobs[1].pid, signal.SIGCONT)
    until(lambda: jobs[1].status()["state"] == "running")
    evidence["checks"].append("external SIGSTOP/SIGCONT detected")
    manager.pause()
    until(lambda: all(job.status()["state"] == "suspended" for job in jobs))
    manager.resume()
    evidence["checks"].append("parent controls suspend/resume all owned children")
    jobs[2].pause()
    until(lambda: jobs[2].status()["state"] == "suspended")
    jobs[2].kill()
    assert jobs[2].status()["state"] == "cancelled"
    evidence["checks"].append("MPS-aware cancellation of a suspended child")
    code = "from mps_runner import connect_mps_manager; m=connect_mps_manager(title='Live MPS acceptance',state_dir=%r); j=m.submit(%r,args=['--seconds','0.4','--value','99'],title='second launcher'); print(j.id)" % (str(state_dir), str(FIXTURE))
    added = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    evidence["second_launcher_job"] = added.stdout.strip()
    failed = manager.submit(FIXTURE, args=["--seconds", "0.3", "--fail"], title="intentional failure")
    results = manager.wait(timeout=30)
    assert failed.status()["state"] == "failed"
    assert removed.status()["state"] == "removed"
    assert sum(job["state"] == "succeeded" for job in results) == 5
    assert len(results) == 8
    time.sleep(0.3)
    assert len(manager.status()["jobs"]) == 8
    evidence["checks"].append("cross-process append, FIFO limits, queue edits, failure isolation, no automatic retry")
    for job in jobs:
        if job.status()["state"] == "succeeded":
            output = json.loads((job.run_dir / "result.json").read_text())
            assert output["pid"] == job.pid
    evidence["checks"].append("real CUDA kernel outputs and displayed child PIDs verified")
    assert any(parent["pid"] == manager.pid for parent in list_mps_managers(state_dir=state_dir))
    evidence["checks"].append("completed parent remains discoverable")
    manager.terminate()
    assert not any(parent["pid"] == manager.pid for parent in list_mps_managers(state_dir=state_dir))
    evidence["checks"].append("explicit terminate persistently dismisses parent and keeps artifacts")
    evidence["passed"] = True
finally:
    try:
        if manager.status()["state"] != "closed":
            manager.shutdown()
    except Exception:
        pass
    (state_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
