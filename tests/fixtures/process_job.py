"""Bounded process-control fixture; all outputs go to its private run directory."""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=0.2)
parser.add_argument("--fail", action="store_true")
parser.add_argument("--label", default="fixture")
parser.add_argument("--stiffness", type=float)
parser.add_argument("--density", type=float)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
directory = args.output or Path(os.environ["MPS_RUN_DIR"])
directory.mkdir(parents=True, exist_ok=True)
print(json.dumps({"pid": os.getpid(), "label": args.label, "args": vars(args) | {"output": str(args.output)}}), flush=True)
started = time.monotonic()
ticks = 0
while time.monotonic() - started < args.seconds:
    ticks += 1
    (directory / "ticks.txt").write_text(str(ticks))
    time.sleep(0.03)
if args.fail:
    raise ValueError("deliberate ordinary script failure")
print("completed", flush=True)
