"""Bounded shared-service and round-robin acceptance on the approved GPU 0."""
import json
import os
from pathlib import Path
import time
import uuid

from mps_runner import run_mps_scripts, list_mps_managers, _start_identity
from test_manager import until
from test_tui import Terminal

root = Path.cwd() / ".mps-manager" / "validation" / ("shared-" + uuid.uuid4().hex[:8])
root.mkdir(parents=True)
fixture = Path(__file__).parent / "fixtures" / "cuda_job.py"
evidence = {"host": os.uname().nodename, "checks": []}
parents, terminals = [], []

try:
    a = run_mps_scripts(gpu=0, gpu_max_concurrent=1, max_concurrent=2,
        title="SHARED GPU A", state_dir=root, detach=True, common_args=["--value", "17"])
    parents.append(a)
    blocker = a.submit(fixture, args=["--seconds", "12"], title="blocker")
    until(lambda: blocker.status()["state"] == "running", timeout=25)
    # No pipe paths or references to A: automatic same-GPU sharing is the default.
    b = run_mps_scripts(gpu=0, max_concurrent=2, title="SHARED GPU B", state_dir=root, detach=True)
    parents.append(b)
    sa, sb = a.status(), b.status()
    assert sa["service_pid"] == sb["service_pid"]
    assert sa["mps_pipe_directory"] == sb["mps_pipe_directory"]
    assert sb["gpu_max_concurrent"] == 1
    evidence.update(service_pid=sa["service_pid"], pipe=sa["mps_pipe_directory"], gpu_uuid=sa["gpu_uuid"], parents=[a.pid,b.pid])
    evidence["checks"].append("parents automatically share one verified MPS endpoint and inherit GPU limit")
    aa = [a.submit(fixture, args=["--seconds", "0.25"], title=f"A{i}") for i in range(3)]
    bb = [b.submit(fixture, args=["--seconds", "0.25"], title=f"B{i}") for i in range(3)]
    blocker.pause()
    until(lambda: blocker.status()["state"] == "suspended")
    time.sleep(0.15)
    assert all(j.pid is None for j in aa+bb)
    terminal = Terminal(root)
    terminals.append(terminal)
    terminal.wait(lambda text: "SHARED GPU A" in text and "SHARED GPU B" in text)
    (root / "both-parents.txt").write_text(terminal.text)
    evidence["checks"].append("shared parents remain separate TUI entries; suspended client retains shared slot")
    blocker.kill()
    a.wait(timeout=20)
    b.wait(timeout=20)
    ordered = sorted([j.status() for j in aa+bb], key=lambda j:j["admission_sequence"])
    assert [j["title"] for j in ordered] == ["B0","A0","B1","A1","B2","A2"]
    evidence["admission_order"] = [j["title"] for j in ordered]
    evidence["checks"].append("round-robin admission across parents, FIFO within each parent")
    for job in aa:
        assert json.loads((job.run_dir/"result.json").read_text())["value"] == 17
        assert job.status()["common_args"] == ["--value","17"]
    evidence["checks"].append("common arguments reach real CUDA kernels and are recorded in child metadata")
    swept = b.submit_sweep(fixture, parameters={"value":29,"seconds":{"$sweep":[0.2,0.3]}},
        common_args=["--value","29"], make_args=lambda p,out:["--seconds",str(p["seconds"])])
    b.wait(timeout=15)
    assert all(j.status()["parameters"]["value"] == 29 for j in swept)
    assert all(json.loads((j.run_dir/"result.json").read_text())["value"] == 29 for j in swept)
    survivor = b.submit(fixture,args=["--seconds","3"],title="B SURVIVES")
    until(lambda: survivor.status()["state"] == "running")
    # Newest B first; select A, terminate it with the actual human-facing hotkey.
    terminal.keys(b"\x1b[Bt")
    terminal.wait(lambda text: "SHARED GPU A" not in text and "SHARED GPU B" in text)
    assert survivor.status()["state"] == "running"
    assert not any(p["pid"] == a.pid for p in list_mps_managers(state_dir=root))
    b.wait(timeout=15)
    assert survivor.status()["state"] == "succeeded"
    evidence["checks"].append("t terminates the first parent independently; the shared service and other parent's CUDA job survive")
    service_start = _start_identity(sa["service_pid"])
    b.terminate()
    until(lambda: _start_identity(sa["service_pid"]) != service_start)
    evidence["checks"].append("last parent closes the shared MPS service")
    # A new session can choose a different GPU-wide limit, and use it concurrently.
    c = run_mps_scripts(gpu=0, gpu_max_concurrent=2, title="Parallel C", state_dir=root, detach=True)
    parents.append(c)
    jc = c.submit(fixture,args=["--seconds","4"])
    until(lambda: jc.status()["state"] == "running")
    d = run_mps_scripts(gpu=0, title="Parallel D", state_dir=root, detach=True)
    parents.append(d)
    jd = d.submit(fixture,args=["--seconds","4"])
    until(lambda: jc.status()["state"] == jd.status()["state"] == "running")
    assert c.status()["mps_pipe_directory"] == d.status()["mps_pipe_directory"]
    assert d.status()["gpu_max_concurrent"] == 2
    c.terminate()
    assert jd.status()["state"] == "running"
    d.wait(timeout=15)
    assert jd.status()["state"] == "succeeded"
    final_service = d.status()["service_pid"]
    final_start = _start_identity(final_service)
    d.terminate()
    until(lambda: _start_identity(final_service) != final_start)
    evidence["checks"].append("two parents run real CUDA jobs concurrently in the same MPS service")
    evidence["passed"] = True
finally:
    for terminal in terminals:
        terminal.close()
    for parent in reversed(parents):
        parent.shutdown()
    (root/"evidence.json").write_text(json.dumps(evidence,indent=2))
print(json.dumps(dict(evidence,state_dir=str(root)),indent=2))
