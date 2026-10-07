"""cluster/feed_queue.sh against stub sbatch / squeue: a plan line waits while the queue
has no room, is submitted once there is, is skipped once its dry run has nothing left,
and a --fill line waits until no job of its wave is queued."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "cluster" / "feed_queue.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def wait_for(log: Path, text: str, timeout: float = 40.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if log.exists() and text in log.read_text():
            return log.read_text()
        time.sleep(0.2)
    raise AssertionError(f"'{text}' not seen in {log}:\n{log.read_text() if log.exists() else ''}")


def stubs(tmp_path, queued: int):
    """A stub squeue (task count from a file; job names from another with -o %j), a stub
    sbatch that logs its calls and refuses from the N-th call on if ``fail_from`` holds N,
    a private campaign root and the environment for the feeder (LIMIT 50, 1 s rounds)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    count, names, slog, fail = (tmp_path / n for n in ("queued", "names", "sbatch.log", "fail_from"))
    count.write_text(f"{queued}\n")
    names.write_text("")
    (bindir / "squeue").write_text(
        "#!/bin/bash\n"
        f'if [[ " $* " == *" -o "* ]]; then cat "{names}"; else n=$(cat "{count}"); '
        'for ((i = 0; i < n; i++)); do echo task; done; fi\n')
    (bindir / "sbatch").write_text(
        "#!/bin/bash\n"
        f'n=$(( $(wc -l < "{slog}" 2>/dev/null || echo 0) + 1 ))\n'
        f'if [ -f "{fail}" ] && [ "$n" -ge "$(cat "{fail}")" ]; then\n'
        '  echo "sbatch: error: AssocMaxSubmitJobLimit" >&2; exit 1\nfi\n'
        f'echo "$*" >> "{slog}"\n')
    for f in ("squeue", "sbatch"):
        (bindir / f).chmod(0o755)
    log = tmp_path / "feed.log"                                 # never the repository's own log
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
           "CAMPAIGN_ROOT": str(tmp_path / "campaign"), "LIMIT": "50", "INTERVAL": "1",
           "USER": os.environ.get("USER", "tester"), "FEED_LOG": str(log)}
    return count, names, slog, fail, log, env


def calls(slog: Path) -> int:
    return len(slog.read_text().splitlines()) if slog.exists() else 0


def test_feeder_waits_for_room_submits_once_and_guards_fills(tmp_path):
    count, names, slog, _, log, env = stubs(tmp_path, queued=45)  # 45 of 50 queued: room for 5
    root = tmp_path / "campaign"
    plan = tmp_path / "plan.txt"
    plan.write_text("# comment\n--wave 2 --n 1e4\n\n--wave 2 --n 1e4 --fill --time 24:00:00\n")
    proc = subprocess.Popen(["bash", str(FEED), str(plan)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_for(log, "needs 20 tasks, room for 5")               # wave 2 at one n: 20 tasks, waits
        assert calls(slog) == 0
        count.write_text("0\n")
        wait_for(log, "submitted 20 tasks in 3 cells")
        assert calls(slog) == 3
        assert "| bash cluster/submit_campaign.sh --wave 2 --n 1e4 | submitted 20 tasks in 3 cells |" in log.read_text()
        names.write_text("2p10i4\n2p10r4\n")                      # the fill waits for the wave's jobs
        wait_for(log, "jobs of wave 2 still queued or running")
        assert calls(slog) == 3
        names.write_text("")
        for cell, shards in (("search_p10_C2I_n1e4", 2), ("search_p10_Cresc_n1e4", 2), ("search_p20_Cresc_n1e4", 16)):
            (root / cell / "shards").mkdir(parents=True)
            for i in range(shards):                                # every shard written: nothing to fill
                (root / cell / "shards" / f"shard_{i:04d}_of_{shards:04d}.npz").touch()
        wait_for(log, "plan complete")
        assert proc.wait(timeout=20) == 0
        text = log.read_text()
        assert "--fill --time 24:00:00': nothing left to submit, done" in text
        assert calls(slog) == 3                                    # the fill submitted nothing
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_line_larger_than_the_cap_goes_in_parts(tmp_path):
    """Wave 4 at p = 40 has 224 tasks, more than the cap: the feeder runs it whenever there
    is room for its largest cell (32), submit_campaign.sh submits cells until sbatch refuses,
    and the rest follows in a later round; a run that got cells accepted is not an idle run."""
    count, _, slog, fail, log, env = stubs(tmp_path, queued=20)   # room 30 < 32: waits
    plan = tmp_path / "plan.txt"
    plan.write_text("--wave 4 --p 40\n")
    proc = subprocess.Popen(["bash", str(FEED), str(plan)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_for(log, "224 tasks, more than the cap, goes in parts; its largest cell needs 32, room for 30")
        assert calls(slog) == 0
        fail.write_text("3")                                      # the third sbatch call is refused
        count.write_text("10\n")                                  # room 40 >= 32
        wait_for(log, "tasks were accepted")
        assert calls(slog) == 2                                    # lasso (8) and MCP (16) of C2I
        assert "| 24 tasks accepted, then sbatch refused |" in log.read_text()
        fail.unlink()
        wait_for(log, "plan complete")
        assert proc.wait(timeout=20) == 0
        assert calls(slog) == 12 and "submitted 200 tasks in 10 cells" in log.read_text()
        assert "giving up" not in log.read_text()
    finally:
        if proc.poll() is None:
            proc.kill()
