"""Real PTY tests inspect the visible screen, not accumulated escape sequences."""
import errno
import codecs
import fcntl
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest

import pyte

from mps_runner import list_mps_managers, run_mps_scripts

FIXTURE = Path(__file__).parent / "fixtures" / "process_job.py"


class Terminal:
    def __init__(self, root, pid=None):
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 35, 145, 0, 0))
        self.before = termios.tcgetattr(self.slave)
        self.screen = pyte.Screen(145, 35)
        self.stream = pyte.Stream(self.screen)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        args = [sys.executable, "-m", "mps_runner.cli", "tui", "--state-dir", str(root)]
        if pid is not None:
            args += ["--pid", str(pid)]
        self.process = subprocess.Popen(args, stdin=self.slave, stdout=self.slave, stderr=self.slave,
                                        env=dict(os.environ, TERM="xterm-256color"))

    @property
    def text(self):
        return "\n".join(self.screen.display)

    def pump(self, seconds=0.15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if select.select([self.master], [], [], 0.02)[0]:
                try:
                    data = os.read(self.master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                self.stream.feed(self.decoder.decode(data))
                if b"\x1b[6n" in data:
                    os.write(self.master, b"\x1b[1;1R")

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump()
            if predicate(self.text):
                return self.text
        raise AssertionError("unexpected screen:\n" + self.text)

    def keys(self, data):
        os.write(self.master, data)
        self.pump(0.25)

    def close(self):
        if self.process.poll() is None:
            self.keys(b"q")
            self.process.wait(timeout=5)
        assert termios.tcgetattr(self.slave) == self.before, "terminal attributes were not restored"
        os.close(self.master)
        os.close(self.slave)


class TuiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wf-manager-tui-")
        self.root = Path(self.temporary.name) / ".mps-manager"
        self.a = run_mps_scripts(mps=False, detach=True, title="QUEUE ALPHA", state_dir=self.root, max_concurrent=1)
        self.a.pause()
        self.job = self.a.submit(FIXTURE, args=["--seconds", "0.2", "--label", "LOG MARKER"], title="CHILD ONLY LABEL")
        self.b = run_mps_scripts(mps=False, detach=True, title="QUEUE BETA", state_dir=self.root)
        self.terminals = []

    def tearDown(self):
        for terminal in self.terminals:
            terminal.close()
        for parent in (self.a, self.b):
            try:
                parent.shutdown()
            except Exception:
                pass
        self.temporary.cleanup()

    def terminal(self, pid=None):
        terminal = Terminal(self.root, pid)
        self.terminals.append(terminal)
        return terminal

    def test_parent_first_navigation_logs_and_detach(self):
        terminal = self.terminal()
        text = terminal.wait(lambda text: "QUEUE ALPHA" in text and "QUEUE BETA" in text)
        self.assertIn("Parent queues", text)
        self.assertNotIn("CHILD ONLY LABEL", text)
        self.assertNotIn("Study", text)
        terminal.keys(b"\x1b[B\r")  # Newest BETA first, select ALPHA, Enter.
        terminal.wait(lambda text: "CHILD ONLY LABEL" in text)
        self.assertIn("queue #", terminal.text)
        terminal.keys(b":resume-all\r")
        self.a.wait(timeout=6)
        terminal.keys(b"\x1b[C")
        terminal.wait(lambda text: "Job details" in text)
        self.assertIn("PID (exited)", terminal.text)
        terminal.keys(b"\x1b[C\t")
        terminal.wait(lambda text: "LOG MARKER" in text and "stdout" in text)
        terminal.keys(b"\x1b[D\x1b[D\x1b[D")
        terminal.wait(lambda text: "Parent queues" in text)
        self.assertIn("finished", terminal.text)
        terminal.close()
        self.terminals.remove(terminal)
        self.assertEqual(self.a.status()["state"], "ready", "closing viewer must leave the parent alive")

    def test_direct_jump_and_t_persistently_dismiss_parent(self):
        self.a.resume()
        self.a.wait(timeout=6)
        terminal = self.terminal(self.a.pid)
        terminal.wait(lambda text: "CHILD ONLY LABEL" in text)
        self.assertNotIn("QUEUE BETA", terminal.text)
        terminal.keys(b"t")
        terminal.wait(lambda text: "Parent queues" in text and "QUEUE ALPHA" not in text)
        self.assertIn("QUEUE BETA", terminal.text)
        self.assertFalse(any(parent["pid"] == self.a.pid for parent in list_mps_managers(state_dir=self.root)))
        self.assertTrue((self.job.run_dir.parent / "stdout.log").exists())
        terminal.close()
        self.terminals.remove(terminal)
        reopened = self.terminal()
        text = reopened.wait(lambda text: "QUEUE BETA" in text)
        self.assertNotIn("QUEUE ALPHA", text)

    def test_title_command_does_not_trigger_t_hotkey(self):
        terminal = self.terminal(self.a.pid)
        terminal.wait(lambda text: "CHILD ONLY LABEL" in text)
        terminal.keys(b":title RENAMED QUEUE\r")
        terminal.wait(lambda text: "RENAMED QUEUE" in text)
        self.assertEqual(self.a.title, "RENAMED QUEUE")
        self.assertTrue(any(parent["pid"] == self.a.pid for parent in list_mps_managers(state_dir=self.root)))


if __name__ == "__main__":
    unittest.main()
