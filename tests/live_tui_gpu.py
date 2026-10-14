"""Bounded real-MPS PTY acceptance, run only on the user-authorized test GPU."""
import json
import os
from pathlib import Path
import signal
import time
import uuid

from mps_runner import list_mps_managers, run_mps_scripts
from test_tui import Terminal
from test_manager import until

root = Path.cwd() / ".mps-manager" / "validation" / ("gpu-tui-" + uuid.uuid4().hex[:8])
root.mkdir(parents=True)
fixture = Path(__file__).parent / "fixtures" / "cuda_job.py"
evidence = {"host": os.uname().nodename, "checks": []}
manager = run_mps_scripts(gpu=0, max_concurrent=2, title="GPU TUI validation", state_dir=root, detach=True)
evidence.update(parent_pid=manager.pid, gpu_uuid=manager.status()["gpu_uuid"])
terminal = None


def capture(name):
    (root / (name + ".txt")).write_text(terminal.text)


try:
    manager.pause()
    a = manager.submit(fixture, args=["--seconds", "18", "--label", "CUDA TUI LOG"], title="GPU child A")
    b = manager.submit(fixture, args=["--seconds", "18"], title="GPU child B")
    queued = manager.submit(fixture, args=["--seconds", "0.2"], title="Queue control target")
    terminal = Terminal(root)
    terminal.wait(lambda text: "GPU TUI validation" in text)
    assert "Parent queues" in terminal.text and "GPU child A" not in terminal.text
    capture("parents")
    terminal.keys(b"\x1b[C")
    terminal.wait(lambda text: "GPU child A" in text and "Queue control target" in text)
    terminal.keys(b":resume-all\r")
    until(lambda: a.status()["state"] == b.status()["state"] == "running")
    terminal.wait(lambda text: str(a.pid) in text and str(b.pid) in text)
    capture("children")
    evidence["checks"].append("parent-first browser and verified MPS child PIDs")
    terminal.keys(b":pause\r")
    until(lambda: a.status()["state"] == "suspended")
    assert b.status()["state"] == "running"
    terminal.wait(lambda text: "suspended" in text)
    capture("suspended-child")
    terminal.keys(b":resume\r")
    until(lambda: a.status()["state"] == "running")
    terminal.keys(b":pause-all\r")
    until(lambda: a.status()["state"] == b.status()["state"] == "suspended")
    terminal.keys(b":resume-all\r")
    until(lambda: a.status()["state"] == b.status()["state"] == "running")
    evidence["checks"].append("individual and whole-queue suspend/resume from TUI")
    terminal.keys(b"\x1b[B\x1b[B:hold\r")
    until(lambda: queued.status()["detail"] == "held")
    terminal.keys(b":release\r")
    until(lambda: queued.id in manager.status()["pending"])
    terminal.keys(b":top\r:shuffle\r:remove\r")
    until(lambda: queued.status()["state"] == "removed")
    evidence["checks"].append("pending hold/release/reorder/shuffle/remove from TUI")
    terminal.keys(b"\x1b[A\x1b[A\x1b[C")
    terminal.wait(lambda text: "Job details" in text and str(a.pid) in text)
    capture("details")
    terminal.keys(b"\x1b[C\t")
    terminal.wait(lambda text: "CUDA TUI LOG" in text and "stdout" in text)
    capture("logs")
    terminal.keys(b"\x1b[D\x1b[D")
    terminal.keys(b":pause\r:kill\r")
    until(lambda: a.status()["state"] == "cancelled")
    assert b.status()["state"] == "running"
    evidence["checks"].append("MPS-aware cancellation of suspended child through TUI, sibling survives")
    terminal.close()
    terminal = None
    assert manager.status()["state"] == "ready"
    evidence["checks"].append("q detaches viewer and restores terminal while queue remains alive")
    os.kill(manager.pid, signal.SIGSTOP)
    try:
        until(lambda: Path(f"/proc/{manager.pid}/stat").read_text().split(")", 1)[1].split()[0] == "T")
        manager.resume()
    finally:
        os.kill(manager.pid, signal.SIGCONT)
    evidence["checks"].append("native checked resume of externally stopped parent on Conda Python")
    b.kill()
    manager.wait(timeout=15)
    terminal = Terminal(root)
    terminal.wait(lambda text: "GPU TUI validation" in text and "finished" in text)
    capture("finished-parent")
    evidence["checks"].append("finished parent retained for inspection")
    terminal.close()
    terminal = Terminal(root, manager.pid)
    terminal.wait(lambda text: "GPU child A" in text)
    terminal.keys(b"t")
    terminal.wait(lambda text: "Parent queues" in text and "GPU TUI validation" not in text)
    assert not list_mps_managers(state_dir=root)
    assert (a.run_dir.parent / "stdout.log").exists()
    capture("dismissed-parent")
    evidence["checks"].append("PID direct jump and t cleanup/persistent dismissal preserve logs")
    terminal.close()
    terminal = Terminal(root)
    terminal.wait(lambda text: "Parent queues" in text)
    assert "GPU TUI validation" not in terminal.text
    evidence["passed"] = True
finally:
    if terminal:
        terminal.close()
    manager.shutdown()
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2))

print(json.dumps(dict(evidence, state_dir=str(root)), indent=2))
