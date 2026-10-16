"""Verify automatic launch-time thread limits using real CUDA device attributes."""
import json
import math
from pathlib import Path
import uuid

from mps_runner import _start_identity, run_mps_scripts
from test_manager import until

root = Path.cwd() / ".mps-manager" / "validation" / ("threads-" + uuid.uuid4().hex[:8])
root.mkdir(parents=True)
fixture = Path(__file__).parent / "fixtures" / "cuda_job.py"
parents = []
evidence = {"batches": [], "checks": []}


def batch(queue, count, *, seconds=0.25, env=None):
    jobs = queue.submit_many([
        {"script_path": fixture, "args": ["--seconds", str(seconds)], "env": env or {}}
        for _ in range(count)
    ])
    return jobs


def results(queue, jobs):
    queue.wait(timeout=30)
    assert all(job.status()["state"] == "succeeded" for job in jobs), [job.status() for job in jobs]
    return [json.loads((job.run_dir / "result.json").read_text()) for job in jobs]


try:
    a = run_mps_scripts(gpu=0, max_concurrent=8, title="Thread ceiling A", state_dir=root, detach=True)
    parents.append(a)
    baseline_jobs = batch(a, 1)
    baseline = results(a, baseline_jobs)[0]
    full = baseline["multiprocessors"]
    assert baseline_jobs[0].status()["active_thread_percentage"] == 100
    for count in (2, 3, 4, 8):
        jobs = batch(a, count)
        observed = results(a, jobs)
        expected = min(100, 200 / count)
        for job, result in zip(jobs, observed):
            state = job.status()
            assert state["admission_clients"] == count, state
            assert math.isclose(state["active_thread_percentage"], expected), state
            assert math.isclose(float(result["active_thread_percentage"]), expected), result
            assert 0 < result["multiprocessors"] <= math.ceil(full * expected / 100), result
            if expected < 100:
                assert result["multiprocessors"] < full, result
        evidence["batches"].append({"clients": count, "percentage": expected,
                                    "multiprocessors": [r["multiprocessors"] for r in observed]})
    evidence["checks"].append("1/2/3/4/8-client defaults reach real CUDA contexts and change available SM counts")
    # Limits apply to newly launched clients across parents; existing clients keep theirs.
    a.shutdown()
    a = run_mps_scripts(gpu=0, max_concurrent=4, title="Parent A four", state_dir=root, detach=True)
    parents.append(a)
    b = run_mps_scripts(gpu=0, max_concurrent=4, title="Parent B four", state_dir=root, detach=True)
    parents.append(b)
    aa = batch(a, 4, seconds=20)
    until(lambda: all(j.status()["state"] == "running" for j in aa), timeout=25)
    assert all(j.status()["active_thread_percentage"] == 50 for j in aa)
    bb = batch(b, 4, seconds=0.3)
    observed = results(b, bb)
    assert all(j.status()["admission_clients"] == 8 for j in bb)
    assert all(j.status()["active_thread_percentage"] == 25 for j in bb)
    assert all(r["multiprocessors"] <= math.ceil(full / 4) for r in observed)
    assert all(j.status()["active_thread_percentage"] == 50 for j in aa)
    evidence["checks"].append("new jobs count clients in both parents; existing clients keep their launch-time ceiling")
    for job in aa:
        job.kill()
    custom = batch(b, 1, env={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "37.5"})
    observed = results(b, custom)
    assert custom[0].status()["active_thread_percentage"] == 37.5
    assert float(observed[0]["active_thread_percentage"]) == 37.5
    evidence["checks"].append("existing explicit client environment overrides remain supported")
    evidence["passed"] = True
finally:
    service = parents[-1].status()["service_pid"] if parents else None
    service_start = _start_identity(service) if service else None
    for parent in reversed(parents):
        parent.shutdown()
    if service:
        until(lambda: _start_identity(service) != service_start, timeout=15)
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2))

print(json.dumps(dict(evidence, state_dir=str(root)), indent=2))
