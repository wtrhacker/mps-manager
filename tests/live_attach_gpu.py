"""Verify independently controlled queues can attach the same MPS endpoint."""
import json
import os
from pathlib import Path
import uuid

from mps_runner import MpsManagerError, list_mps_managers, run_mps_scripts
from test_manager import until

root = Path.cwd() / ".mps-manager" / "validation" / ("gpu-attach-" + uuid.uuid4().hex[:8])
root.mkdir(parents=True)
fixture = Path(__file__).parent / "fixtures" / "cuda_job.py"
evidence = {"host": os.uname().nodename, "checks": []}
owner = run_mps_scripts(gpu=0, title="Endpoint owner", state_dir=root, detach=True)
attached = None
try:
    evidence.update(parent_pid=owner.pid, gpu_uuid=owner.status()["gpu_uuid"])
    try:
        run_mps_scripts(gpu=0, state_dir=root, detach=True)
    except MpsManagerError:
        evidence["checks"].append("second private owner fails explicitly")
    else:
        raise AssertionError("second private owner must not silently use the GPU")
    a = owner.submit(fixture, args=["--seconds", "18"], title="owner child")
    attached = run_mps_scripts(gpu=0, title="Attached parent", state_dir=root, detach=True,
                               mps_pipe_directory=owner.status()["mps_pipe_directory"])
    b = attached.submit(fixture, args=["--seconds", "18"], title="attached child")
    until(lambda: a.status()["state"] == b.status()["state"] == "running")
    evidence["attached_parent_pid"] = attached.pid
    a.pause()
    a.kill()
    assert b.status()["state"] == "running"
    evidence["checks"].append("attached endpoint verifies clients; owned child cancellation preserves other parent")
    try:
        owner.terminate()
    except MpsManagerError:
        assert owner.status()["state"] == "fault"
        assert any(parent["pid"] == owner.pid for parent in list_mps_managers(state_dir=root))
        assert b.status()["state"] == "running"
        evidence["checks"].append("owner refuses daemon shutdown with attached clients and remains controllable/visible")
    else:
        raise AssertionError("owner must preserve a service with another parent's live client")
    attached.terminate()
    owner.terminate()
    assert not any(parent["pid"] in {owner.pid, attached.pid} for parent in list_mps_managers(state_dir=root))
    evidence["checks"].append("attached parent never shuts down shared daemon; explicit owner cleanup succeeds afterwards")
    evidence["passed"] = True
finally:
    if attached:
        attached.shutdown()
    owner.shutdown()
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2))
print(json.dumps(dict(evidence, state_dir=str(root)), indent=2))
