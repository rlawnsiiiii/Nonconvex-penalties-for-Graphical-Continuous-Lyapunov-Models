"""The cluster path for the n-sweep: the sample-size option, the pairing across n
that every comparison relies on, the shard runner end to end, and the submit
script's bookkeeping (against a stub ``sbatch``)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gclm.config import S1Config, parse_n_obs
from gclm.data.simulate import CChoice, draw_instance

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "simulations" / "run_s1_shard.py"
SUBMIT = ROOT / "cluster" / "submit_nsweep.sh"


@pytest.mark.parametrize("text, value", [("1000", 1000), ("1e5", 100_000), ("1e4", 10_000),
                                         ("inf", math.inf), ("Inf", math.inf)])
def test_parse_n_obs_accepts_integers_and_inf(text, value):
    assert parse_n_obs(text) == value
    assert isinstance(parse_n_obs(text), float if math.isinf(value) else int)


@pytest.mark.parametrize("text", ["0", "1", "-5", "1.5", "-inf", "nan", "abc"])
def test_parse_n_obs_rejects_bad_values(text):
    with pytest.raises(ValueError):
        parse_n_obs(text)


def test_drift_and_volatility_do_not_depend_on_n():
    """M* and C are drawn before the data, so runs at different n see the same
    models -- the comparisons across the n-sweep are paired dataset by dataset."""
    cfg = S1Config()
    seed = [cfg.seed, 10, 2, list(CChoice).index(CChoice.RANDOM_FULL), 7]
    draws = {n: draw_instance(10, 2, n, CChoice.RANDOM_FULL, np.random.default_rng(seed),
                              standardize=True)
             for n in (200, 1000, math.inf)}
    m_ref, c_ref, s_ref, _ = draws[math.inf]
    for m, c, s, _ in draws.values():
        assert np.array_equal(m, m_ref) and np.array_equal(c, c_ref) and np.array_equal(s, s_ref)
    # n = inf is the standardised population covariance; finite n is a sample
    d = np.sqrt(np.diag(s_ref))
    assert np.allclose(draws[math.inf][3], s_ref / np.outer(d, d))
    assert not np.allclose(draws[1000][3], draws[math.inf][3])


def _run_shard(tmp_path, n_obs, shard=0, n_shards=16):
    out = tmp_path / f"n{n_obs}"
    cmd = [sys.executable, str(RUNNER), "--shard", str(shard), "--n-shards", str(n_shards),
           "--p", "5", "--reps", "1", "--n-obs", n_obs, "--out-dir", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT), out


def test_shard_runner_records_n_and_pairs_datasets_across_n(tmp_path):
    """One dataset (p = 5) at n = inf and at n = 1000: the shard records the n it
    used, and M* is the same in both."""
    shards = {}
    for n_obs in ("inf", "1000"):
        res, out = _run_shard(tmp_path, n_obs)
        assert res.returncode == 0, res.stderr
        (f,) = out.glob("shard_*.npz")
        shards[n_obs] = np.load(f, allow_pickle=True)
    assert json.loads(str(shards["inf"]["config_json"]))["n_obs"] == math.inf
    assert json.loads(str(shards["1000"]["config_json"]))["n_obs"] == 1000
    for key in ("p", "k", "c_choice", "rep", "m_true_i", "m_true_j"):
        assert len(shards["inf"][key]) == 1
    for key in ("m_true_i", "m_true_j", "m_true_v"):
        assert np.array_equal(shards["inf"][key][0], shards["1000"][key][0])
    assert shards["inf"]["lambda_max"][0] != shards["1000"]["lambda_max"][0]


def test_shard_runner_rejects_out_of_range_shard(tmp_path):
    res, out = _run_shard(tmp_path, "inf", shard=16, n_shards=16)
    assert res.returncode != 0 and "--shard" in res.stderr
    assert not list(out.glob("*.npz"))


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_submit_script_submits_once_and_fills_missing_shards(tmp_path):
    """Against a stub sbatch: a first call submits the full array and records the
    shard count; a second call and --status submit nothing; --fill resubmits
    exactly the shards that are missing; a finished cell is reported complete."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    stub = bindir / "sbatch"
    stub.write_text(f'#!/bin/bash\necho "$*" >> "{log}"\n')
    stub.chmod(0o755)
    root = tmp_path / "nsweep"
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
           "NSWEEP_ROOT": str(root)}
    logs_existed = (ROOT / "logs").exists()      # the script creates it for sbatch -o

    def submit(*extra):
        res = subprocess.run(["bash", str(SUBMIT), "--n", "inf", "--loss", "direct",
                              "--penalty", "MCP", *extra],
                             capture_output=True, text=True, env=env, cwd=ROOT)
        assert res.returncode == 0, res.stderr
        return res.stdout, log.read_text().splitlines() if log.exists() else []

    try:
        run = root / "direct_MCP_ninf"
        _, calls = submit()
        assert len(calls) == 1
        assert "--array=0-15" in calls[0] and "--time=06:00:00" in calls[0]
        assert f"{run} --p 10 20 --reps 25 --n-obs inf --loss direct --penalty MCP" in calls[0]
        assert (run / "n_shards").read_text().strip() == "16"

        out, calls = submit()                                   # already submitted
        assert len(calls) == 1 and "0/16 shards written" in out
        out, calls = submit("--status")                         # reports, never submits
        assert len(calls) == 1 and "0/16 shards written" in out

        (run / "s1_shards").mkdir()
        for i in set(range(16)) - {3, 11}:
            (run / "s1_shards" / f"shard_{i:04d}_of_0016.npz").touch()
        out, calls = submit("--status")
        assert len(calls) == 1 and "14/16 shards written" in out
        _, calls = submit("--fill", "--time", "12:00:00")
        assert len(calls) == 2 and "--array=3,11" in calls[1] and "--time=12:00:00" in calls[1]
        assert (run / "n_shards").read_text().strip() == "16"

        for i in (3, 11):
            (run / "s1_shards" / f"shard_{i:04d}_of_0016.npz").touch()
        out, calls = submit("--fill")
        assert len(calls) == 2 and "complete" in out
    finally:
        if not logs_existed and (ROOT / "logs").is_dir() and not any((ROOT / "logs").iterdir()):
            (ROOT / "logs").rmdir()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_submit_script_shards_override_applies_to_new_cells_only(tmp_path):
    """--shards sets the array and the recorded shard count of a new cell; a cell
    submitted before keeps its own count under --fill."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.log"
    (bindir / "sbatch").write_text(f'#!/bin/bash\necho "$*" >> "{log}"\n')
    (bindir / "sbatch").chmod(0o755)
    root = tmp_path / "nsweep"
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "NSWEEP_ROOT": str(root)}

    def submit(*extra):
        res = subprocess.run(["bash", str(SUBMIT), "--n", "1e4", "--loss", "frobenius",
                              "--penalty", "lasso", *extra],
                             capture_output=True, text=True, env=env, cwd=ROOT)
        return res, log.read_text().splitlines() if log.exists() else []

    logs_existed = (ROOT / "logs").exists()
    try:
        res, calls = submit("--shards", "16", "--time", "24:00:00")
        assert res.returncode == 0, res.stderr
        assert "--array=0-15" in calls[0] and "--time=24:00:00" in calls[0]
        run = root / "frobenius_lasso_n1e4"
        assert (run / "n_shards").read_text().strip() == "16"
        res, calls = submit("--fill", "--shards", "4")              # keeps 16: shards 0..15 missing
        assert "--array=" + ",".join(str(i) for i in range(16)) in calls[1]
        assert (run / "n_shards").read_text().strip() == "16"
        res, _ = submit("--shards", "0")
        assert res.returncode == 2 and "positive integer" in res.stderr
    finally:
        if not logs_existed and (ROOT / "logs").is_dir() and not any((ROOT / "logs").iterdir()):
            (ROOT / "logs").rmdir()
