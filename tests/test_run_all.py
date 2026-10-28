import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_all import DEFAULT_ORDER, STEPS, run_steps


def fake_runner(fail_on=None):
    calls = []

    def run(cmd, cwd, env):
        calls.append(cmd[2])
        return subprocess.CompletedProcess(cmd, 1 if cmd[2] == fail_on else 0)

    return run, calls


def test_default_order_only_known_steps():
    assert set(DEFAULT_ORDER) <= set(STEPS)


def test_runs_steps_in_given_order():
    run, calls = fake_runner()
    assert run_steps(["robustness", "business"], runner=run) == 0
    assert calls == ["src.robustness", "src.business"]


def test_stops_at_first_failure():
    run, calls = fake_runner(fail_on="src.business")
    assert run_steps(["robustness", "business", "segmentation"], runner=run) == 1
    assert calls == ["src.robustness", "src.business"]


def test_unknown_step_rejected_before_running():
    run, calls = fake_runner()
    assert run_steps(["robustness", "nope"], runner=run) == 2
    assert calls == []
