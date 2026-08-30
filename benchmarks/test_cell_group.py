import ast
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cell_group


HERE = Path(__file__).resolve().parent


class CellGroupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def python(self, source, *args):
        return [sys.executable, "-c", source, *map(str, args)]

    def wait_file(self, path, timeout=10):
        deadline = time.monotonic() + timeout
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(path.exists(), f"child did not publish {path}")

    def alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        if Path(f"/proc/{pid}/stat").exists():
            with contextlib.suppress(OSError, IndexError):
                return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        return True

    def wait_gone(self, pid, timeout=10):
        deadline = time.monotonic() + timeout
        while self.alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        return not self.alive(pid)

    def test_a_sleep_chain_is_killed_and_reaped_as_one_group(self):
        pidfile = self.dir / "grandchild.pid"
        child = cell_group.spawn(
            self.python(
                "import os, signal, subprocess, sys, time\n"
                "code = ('import signal, time; '\n"
                "        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)')\n"
                "kid = subprocess.Popen([sys.executable, '-c', code],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "open(sys.argv[1], 'w').write(str(kid.pid))\n"
                "print('partial child output', flush=True)\n"
                "time.sleep(300)\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.wait_file(pidfile)
        grandchild = int(pidfile.read_text())

        result = cell_group.terminate(child, grace=0.2)

        self.assertIs(result.observation, cell_group.GONE)
        self.assertTrue(result.reaped)
        self.assertIn("partial child output", result.stdout)
        self.assertTrue(self.wait_gone(grandchild), f"pid {grandchild} survived group cleanup")

    def test_an_orphan_spawner_is_observed_and_collected_after_its_leader_exits(self):
        pidfile = self.dir / "orphan.pid"
        child = cell_group.spawn(
            self.python(
                "import subprocess, sys\n"
                "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "open(sys.argv[1], 'w').write(str(kid.pid))\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        child.communicate(timeout=10)
        self.wait_file(pidfile)
        orphan = int(pidfile.read_text())
        self.addCleanup(lambda: self.alive(orphan) and os.kill(orphan, signal.SIGKILL))

        self.assertIsNot(child.observe(), cell_group.GONE)
        result = cell_group.terminate(child, grace=0.2)

        self.assertIs(result.observation, cell_group.GONE)
        self.assertTrue(self.wait_gone(orphan), f"pid {orphan} survived orphan cleanup")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "zombie census needs procfs")
    def test_a_zombie_only_group_is_observed_gone(self):
        child = cell_group.spawn(self.python("pass"), stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        observed = child.observe()
        while observed is not cell_group.GONE and time.monotonic() < deadline:
            time.sleep(0.02)
            observed = child.observe()
        os.kill(child.pid, 0)  # the unreaped process-table entry still exists

        self.assertIs(observed, cell_group.GONE)
        child.wait()

    def test_sweep_drivers_have_no_unmanaged_spawn_site(self):
        offenders = []
        for path in (HERE / "orchestrate.py", HERE / "gh675_cells.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if (
                    isinstance(owner, ast.Name)
                    and owner.id == "subprocess"
                    and node.func.attr in ("Popen", "run", "call", "check_call", "check_output")
                ):
                    offenders.append(f"{path.name}:{node.lineno} subprocess.{node.func.attr}")
        self.assertEqual(offenders, [], "unmanaged benchmark child sites: " + ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
