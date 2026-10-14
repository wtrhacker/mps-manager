"""One terminal command opens the parent-queue browser."""
import argparse
import json
import os
import sys

from . import MpsManagerError, _binary, connect_mps_manager, list_mps_managers


def main():
    parser = argparse.ArgumentParser(prog="mps-manager")
    sub = parser.add_subparsers(dest="command", required=True)
    tui = sub.add_parser("tui", help="browse titled parent queues, jobs, and logs")
    tui.add_argument("--pid", type=int, help="jump directly into one parent queue")
    tui.add_argument("--state-dir", default=".mps-manager")
    listing = sub.add_parser("list", help="list parent queues as JSON")
    listing.add_argument("--state-dir", default=".mps-manager")
    for name in ("status", "shutdown"):
        child = sub.add_parser(name)
        child.add_argument("--pid", type=int)
        child.add_argument("--title")
        child.add_argument("--state-dir", default=".mps-manager")
    args = parser.parse_args()
    try:
        _run(args)
    except MpsManagerError as error:
        parser.exit(2, f"mps-manager: {error}\n")


def _run(args):
    if args.command == "tui":
        command = [str(_binary()), "tui", "--state-dir", args.state_dir]
        if args.pid is not None:
            command += ["--pid", str(args.pid)]
        os.execv(command[0], command)
    elif args.command == "list":
        print(json.dumps(list_mps_managers(state_dir=args.state_dir), indent=2))
    else:
        manager = connect_mps_manager(args.pid, title=args.title, state_dir=args.state_dir)
        if args.command == "shutdown":
            manager.shutdown()
        else:
            print(json.dumps(manager.status(), indent=2))


if __name__ == "__main__":
    main()
