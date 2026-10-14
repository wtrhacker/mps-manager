"""Run this file under any name on the GPU host after installing the wheel."""
from mps_runner import run_mps_scripts

manager = run_mps_scripts(
    [
        {"script_path": "simulate.py", "args": ["--stiffness", str(value)], "title": f"stiffness={value}"}
        for value in [0.5, 1.5, 4.0, 8.0]
    ],
    title="Polymer stiffness queue",
    gpu=0,
    max_concurrent=4,
    detach=True,
)
print(f"Queue '{manager.title}' is ready. Open it with: mps-manager tui")
