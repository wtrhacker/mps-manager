"""Integration tests exercise real native managers and real child processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from mps_runner import MpsManagerError, connect_mps_manager, expand_parameters, list_mps_managers, run_mps_scripts

FIXTURE = Path(__file__).parent / "fixtures" / "process_job.py"


def until(function, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = function()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("condition did not become true")


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wf-manager-test-")
        self.root = Path(self.temporary.name) / ".mps-manager"
        self.parents = []

    def tearDown(self):
        for parent in reversed(self.parents):
            try:
                parent.shutdown()
            except MpsManagerError:
                pass
        self.temporary.cleanup()

    def manager(self, **options):
        parent = run_mps_scripts(mps=False, detach=True, state_dir=self.root, title="Test queue", **options)
        self.parents.append(parent)
        return parent

    def test_concurrency_append_and_real_pid(self):
        parent = self.manager(max_concurrent=2)
        jobs = [parent.submit(FIXTURE, args=["--seconds", "1", "--label", str(index)]) for index in range(6)]
        until(lambda: sum(job["state"] == "running" for job in parent.status()["jobs"]) == 2)
        first = parent.status()
        self.assertEqual(sum(job["state"] == "pending" for job in first["jobs"]), 4)
        self.assertEqual(len({job["pid"] for job in first["jobs"] if job["pid"]}), 2)
        other = connect_mps_manager(title="Test queue", state_dir=self.root)
        appended = other.submit(FIXTURE, args=["--label", "later"])
        results = parent.wait(timeout=12)
        self.assertEqual(len(results), 7)
        self.assertTrue(all(job["state"] == "succeeded" for job in results))
        for job in jobs + [appended]:
            record = json.loads(job.logs("stdout").splitlines()[0])
            self.assertEqual(record["pid"], job.pid)
        self.assertEqual(len({str(job.run_dir) for job in jobs + [appended]}), 7)

    def test_second_python_program_connects_by_title(self):
        parent = self.manager(max_concurrent=1)
        code = "from mps_runner import connect_mps_manager; m=connect_mps_manager(title='Test queue',state_dir=%r); j=m.submit(%r,args=['--label','second launcher']); print(j.id)" % (str(self.root), str(FIXTURE))
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=os.environ.copy(), check=True)
        self.assertTrue(result.stdout.strip().startswith("job-"))
        self.assertEqual(parent.wait(timeout=8)[0]["state"], "succeeded")

    def test_failure_is_not_retried_and_siblings_finish(self):
        parent = self.manager(max_concurrent=2)
        failed = parent.submit(FIXTURE, args=["--fail"])
        successful = parent.submit(FIXTURE)
        results = parent.wait(timeout=8)
        self.assertEqual(failed.status()["state"], "failed")
        self.assertEqual(successful.status()["state"], "succeeded")
        time.sleep(0.3)
        self.assertEqual(len(parent.status()["jobs"]), 2)
        self.assertEqual(failed.pid, failed.status()["pid"])
        self.assertIn("deliberate ordinary script failure", failed.logs())
        self.assertEqual([job["exit_code"] for job in results], [1, 0])

    def test_hold_remove_move_shuffle_and_queue_pause(self):
        parent = self.manager(max_concurrent=1)
        parent.pause()
        jobs = [parent.submit(FIXTURE, title=f"job {index}") for index in range(5)]
        self.assertTrue(all(job.pid is None for job in jobs))
        jobs[0].pause()
        self.assertEqual(jobs[0].status()["detail"], "held")
        jobs[1].remove()
        self.assertEqual(jobs[1].status()["state"], "removed")
        parent.move(jobs[4].id, 0)
        self.assertEqual(parent.status()["pending"][0], jobs[4].id)
        before = set(parent.status()["pending"])
        parent.shuffle()
        self.assertEqual(set(parent.status()["pending"]), before)
        parent.reorder([jobs[3].id, jobs[2].id, jobs[4].id])
        self.assertEqual(parent.status()["pending"], [jobs[3].id, jobs[2].id, jobs[4].id])
        jobs[0].resume()
        parent.resume()
        results = parent.wait(timeout=8)
        self.assertEqual(sum(job["state"] == "succeeded" for job in results), 4)
        self.assertEqual(sum(job["state"] == "removed" for job in results), 1)

    def test_individual_external_signals_and_batch_suspend(self):
        parent = self.manager(max_concurrent=2)
        a = parent.submit(FIXTURE, args=["--seconds", "4"])
        b = parent.submit(FIXTURE, args=["--seconds", "4"])
        until(lambda: a.status()["state"] == b.status()["state"] == "running")
        until(lambda: (a.run_dir / "ticks.txt").exists() and (b.run_dir / "ticks.txt").exists())
        a.pause()
        until(lambda: a.status()["state"] == "suspended")
        time.sleep(0.12)
        frozen = (a.run_dir / "ticks.txt").read_text()
        moving = (b.run_dir / "ticks.txt").read_text()
        time.sleep(0.2)
        self.assertEqual((a.run_dir / "ticks.txt").read_text(), frozen)
        self.assertNotEqual((b.run_dir / "ticks.txt").read_text(), moving)
        os.kill(a.pid, signal.SIGCONT)
        until(lambda: a.status()["state"] == "running")
        os.kill(b.pid, signal.SIGSTOP)
        until(lambda: b.status()["state"] == "suspended")
        b.resume()
        parent.pause()
        until(lambda: all(job["state"] == "suspended" for job in parent.status()["jobs"]))
        self.assertTrue(parent.status()["admission_paused"])
        parent.resume()
        a.kill()
        self.assertEqual(a.status()["state"], "cancelled")
        self.assertEqual(b.status()["pid"], b.pid)
        parent.wait(timeout=10)
        self.assertEqual(b.status()["state"], "succeeded")

    def test_native_sweeps_correlated_ranges_replicates_and_no_seed_magic(self):
        self.assertEqual(expand_parameters({"x": {"linspace": {"start": 0, "stop": 1, "num": 3}}}), [{"x": 0.0}, {"x": 0.5}, {"x": 1.0}])
        self.assertEqual(expand_parameters({"$cases": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}), [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        parent = self.manager(max_concurrent=2)
        seen = []
        def args(case, run_dir):
            seen.append((dict(case), case.replicate, run_dir))
            return ["--stiffness", str(case["stiffness"]), "--density", str(case["density"]), "--output", str(run_dir)]
        jobs = parent.submit_sweep(FIXTURE, parameters={"stiffness": {"$sweep": [0.5, 1.5]}, "density": {"$sweep": [0.1, 0.2]}}, make_args=args, replicates=2)
        self.assertEqual(len(jobs), 8)
        self.assertEqual({replicate for _, replicate, _ in seen}, {0, 1})
        self.assertEqual(len({directory for _, _, directory in seen}), 8)
        self.assertTrue(all(set(values) == {"stiffness", "density"} for values, _, _ in seen))
        self.assertTrue(all(job["state"] == "succeeded" for job in parent.wait(timeout=12)))

    def test_sweep_without_callback_passes_fixed_values_and_unique_output_folders(self):
        parent = self.manager(max_concurrent=2, common_args=["--label", "fixed label"])
        jobs = parent.submit_sweep(FIXTURE,
            parameters={"stiffness": {"$sweep": [0.5, 1.5]}, "density": 0.2, "seconds": 0.01},
            output_arg="--output", replicates=2)
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(job["state"] == "succeeded" for job in parent.wait(timeout=8)))
        records = [json.loads(job.logs("stdout").splitlines()[0])["args"] for job in jobs]
        self.assertEqual(sorted(record["stiffness"] for record in records), [0.5, 0.5, 1.5, 1.5])
        self.assertTrue(all(record["density"] == 0.2 and record["label"] == "fixed label" for record in records))
        self.assertEqual(len({record["output"] for record in records}), 4)
        for job, record in zip(jobs, records):
            self.assertEqual(record["output"], str(job.run_dir))
            self.assertTrue((job.run_dir / "ticks.txt").is_file())
            self.assertEqual(job.status()["parameters"]["density"], 0.2)
            self.assertEqual(job.status()["args"].count("--density"), 1)

    def test_sweep_without_output_option_uses_script_arguments_literally(self):
        parent = self.manager()
        job, = parent.submit_sweep(FIXTURE, parameters={"label": "literal space; $(text)", "seconds": 0.01})
        self.assertEqual(parent.wait(timeout=8)[0]["state"], "succeeded")
        record = json.loads(job.logs("stdout").splitlines()[0])["args"]
        self.assertEqual(record["label"], "literal space; $(text)")
        self.assertEqual(record["output"], "None")
        self.assertTrue((job.run_dir / "ticks.txt").exists())

    def test_automatic_sweep_rejects_ambiguous_values_and_duplicate_output(self):
        parent = self.manager()
        for value in (True, None, [1, 2], {"nested": 1}):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "make_args"):
                parent.submit_sweep(FIXTURE, parameters={"label": value})
        with self.assertRaisesRegex(ValueError, "already in parameters"):
            parent.submit_sweep(FIXTURE, parameters={"output": "fixed"}, output_arg="--output")
        with self.assertRaisesRegex(ValueError, "output_arg"):
            parent.submit_sweep(FIXTURE, parameters={"label": "x"}, output_arg="--output bad")
        self.assertEqual(parent.status()["jobs"], [])

    def test_externally_stopped_parent_can_be_resumed_by_native_identity_check(self):
        parent = self.manager(max_concurrent=1)
        parent.pause()
        job = parent.submit(FIXTURE, args=["--seconds", "0.2"])
        os.kill(parent.pid, signal.SIGSTOP)
        try:
            until(lambda: Path(f"/proc/{parent.pid}/stat").read_text().split(")", 1)[1].split()[0] == "T")
            parent.resume()
            self.assertEqual(parent.wait(timeout=8)[0]["state"], "succeeded")
            self.assertIsNotNone(job.pid)
        finally:
            if Path(f"/proc/{parent.pid}").exists():
                os.kill(parent.pid, signal.SIGCONT)

    def test_closed_history_can_be_hidden_without_deleting_outputs(self):
        parent = self.manager()
        job = parent.submit(FIXTURE)
        parent.wait(timeout=8)
        parent.shutdown()
        self.assertTrue(any(item["pid"] == parent.pid for item in list_mps_managers(state_dir=self.root)))
        parent.terminate()
        self.assertFalse(list_mps_managers(state_dir=self.root))
        self.assertTrue((job.run_dir.parent / "stdout.log").exists())
        self.assertEqual(job.status()["state"], "succeeded")
        self.assertEqual(parent.status()["state"], "closed")
        self.assertIn(str(job.pid), job.logs("stdout"))

    def test_titles_discovery_and_invalid_requests_do_not_corrupt_queue(self):
        parent = self.manager(max_concurrent=1)
        parent.set_title("Human title")
        self.assertEqual(connect_mps_manager(title="Human title", state_dir=self.root).pid, parent.pid)
        self.assertEqual(list_mps_managers(state_dir=self.root)[0]["title"], "Human title")
        parent.pause()
        job = parent.submit(FIXTURE)
        with self.assertRaises(MpsManagerError):
            parent.reorder([job.id, job.id])
        self.assertEqual(parent.status()["pending"], [job.id])
        with self.assertRaises(MpsManagerError):
            parent.submit(FIXTURE, env={"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(len(parent.status()["jobs"]), 1)

    def test_missing_mps_raises_before_any_scientific_script(self):
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = str(Path(self.temporary.name) / "empty-path")
        try:
            with self.assertRaises(MpsManagerError):
                run_mps_scripts([{"script_path": FIXTURE}], state_dir=self.root, detach=True, mps=True)
        finally:
            os.environ["PATH"] = original
        self.assertFalse(list(self.root.glob("queues/*/jobs/*/stdout.log")))

    def test_shared_round_robin_fifo_and_independent_termination(self):
        a = self.manager(max_concurrent=1, gpu_max_concurrent=1)
        blocker = a.submit(FIXTURE, args=["--seconds", "10"], title="blocker")
        until(lambda: blocker.status()["state"] == "running")
        b = run_mps_scripts(mps=False, detach=True, state_dir=self.root, title="Parent B", max_concurrent=1, gpu_max_concurrent=1)
        self.parents.append(b)
        self.assertEqual(a.status()["service_pid"], b.status()["service_pid"])
        jobs_a = [a.submit(FIXTURE, args=["--seconds", "0.15"], title=f"A{i}") for i in range(3)]
        jobs_b = [b.submit(FIXTURE, args=["--seconds", "0.15"], title=f"B{i}") for i in range(3)]
        blocker.pause()
        until(lambda: blocker.status()["state"] == "suspended")
        time.sleep(0.15)
        self.assertTrue(all(job.pid is None for job in jobs_a + jobs_b))
        blocker.kill()
        a.wait(timeout=10)
        b.wait(timeout=10)
        order = sorted([j.status() for j in jobs_a + jobs_b], key=lambda j: j["admission_sequence"])
        self.assertEqual([j["title"] for j in order], ["B0", "A0", "B1", "A1", "B2", "A2"])
        running = b.submit(FIXTURE, args=["--seconds", "0.8"])
        until(lambda: running.status()["state"] == "running")
        a.terminate()
        self.assertEqual(running.status()["state"], "running")
        b.wait(timeout=8)
        self.assertEqual(running.status()["state"], "succeeded")
        self.assertEqual([p["title"] for p in list_mps_managers(state_dir=self.root) if p["alive"]], ["Parent B"])

    def test_shared_pause_hold_reorder_and_global_limit(self):
        a = self.manager(max_concurrent=2, gpu_max_concurrent=1)
        blocker = a.submit(FIXTURE, args=["--seconds", "10"])
        until(lambda: blocker.status()["state"] == "running")
        b = run_mps_scripts(mps=False, detach=True, state_dir=self.root, title="Other", max_concurrent=2, gpu_max_concurrent=1)
        self.parents.append(b)
        aa = [a.submit(FIXTURE, title=f"A{i}") for i in range(2)]
        bb = [b.submit(FIXTURE, title=f"B{i}") for i in range(3)]
        bb[0].pause()
        b.move(bb[2].id, 0)
        a.pause()
        blocker.kill()
        b.wait(timeout=8)
        self.assertTrue(all(j.pid is None for j in aa))
        self.assertIsNone(bb[0].pid)
        self.assertLess(bb[2].status()["admission_sequence"], bb[1].status()["admission_sequence"])
        a.resume()
        bb[0].resume()
        a.wait(timeout=8)
        b.wait(timeout=8)
        with self.assertRaises(MpsManagerError):
            run_mps_scripts(mps=False, detach=True, state_dir=self.root, gpu_max_concurrent=2)

    def test_common_args_inheritance_replacement_snapshots_and_fixed_parameters(self):
        parent = self.manager(common_args=["--label", "fixed", "--density", "0.2"])
        parent.pause()
        inherited = parent.submit(FIXTURE, args=["--seconds", "0.1"])
        opted_out = parent.submit(FIXTURE, args=["--label", "separate"], common_args=[])
        swept = parent.submit_sweep(FIXTURE,
            parameters={"stiffness": {"$sweep": [0.5, 1.5]}, "fixed_value": 72},
            common_args=["--density", "0.4"],
            make_args=lambda case, output: ["--stiffness", str(case["stiffness"])])
        connected = connect_mps_manager(parent.pid, state_dir=self.root)
        connected.set_common_args(["--label", "future"])
        future = connected.submit(FIXTURE)
        self.assertEqual(inherited.status()["common_args"], ["--label", "fixed", "--density", "0.2"])
        self.assertEqual(inherited.status()["args"], inherited.status()["common_args"] + ["--seconds", "0.1"])
        self.assertEqual(opted_out.status()["common_args"], [])
        self.assertEqual(future.status()["args"], ["--label", "future"])
        self.assertEqual(swept[0].status()["parameters"], {"stiffness": 0.5, "fixed_value": 72})
        self.assertEqual(swept[1].status()["common_args"], ["--density", "0.4"])
        parent.resume()
        self.assertTrue(all(j["state"] == "succeeded" for j in parent.wait(timeout=10)))
        self.assertIn('"label": "fixed"', inherited.logs("stdout"))
        self.assertIn('"label": "separate"', opted_out.logs("stdout"))
        self.assertIn('"density": 0.4', swept[0].logs("stdout"))

    def test_departed_parent_does_not_reset_round_robin_cursor(self):
        a = self.manager(gpu_max_concurrent=1)
        b = run_mps_scripts(mps=False, detach=True, state_dir=self.root, title="B", gpu_max_concurrent=1)
        c = run_mps_scripts(mps=False, detach=True, state_dir=self.root, title="C", gpu_max_concurrent=1)
        self.parents.extend([b, c])
        blocker = b.submit(FIXTURE, args=["--seconds", "10"])
        until(lambda: blocker.status()["state"] == "running")
        ja = a.submit(FIXTURE)
        jc = c.submit(FIXTURE)
        b.terminate()
        a.wait(timeout=8)
        c.wait(timeout=8)
        self.assertLess(jc.status()["admission_sequence"], ja.status()["admission_sequence"])


if __name__ == "__main__":
    unittest.main()
