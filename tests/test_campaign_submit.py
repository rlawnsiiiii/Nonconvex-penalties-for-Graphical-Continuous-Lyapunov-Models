"""cluster/submit_campaign.sh against a stub ``sbatch``: which cells a wave has,
what is submitted for each, and the bookkeeping that makes the script safe to
rerun (one submission per cell, --status, --fill, stopping when sbatch refuses).
Also: every cell's arguments are accepted by the runner it is sent to."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "cluster" / "submit_campaign.sh"
sys.path.insert(0, str(ROOT / "simulations"))

import rescore_shard  # noqa: E402
import run_s1_shard  # noqa: E402
import run_search_shard  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def cluster(tmp_path):
    """A stub sbatch that logs its arguments (and fails from its N-th call on if
    the file ``fail_from`` holds N), and a private campaign root."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log, fail = tmp_path / "sbatch.log", tmp_path / "fail_from"
    stub = bindir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        f'n=$(( $(wc -l < "{log}" 2>/dev/null || echo 0) + 1 ))\n'
        f'if [ -f "{fail}" ] && [ "$n" -ge "$(cat "{fail}")" ]; then\n'
        '  echo "sbatch: error: AssocMaxSubmitJobLimit" >&2; exit 1\nfi\n'
        f'echo "$*" >> "{log}"\n')
    stub.chmod(0o755)
    root = tmp_path / "campaign"
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
           "CAMPAIGN_ROOT": str(root)}
    logs_existed = (ROOT / "logs").exists()            # the script creates it for sbatch -o

    def submit(*args):
        res = subprocess.run(["bash", str(SUBMIT), *args], capture_output=True, text=True,
                             env=env, cwd=ROOT)
        return res, log.read_text().splitlines() if log.exists() else []

    yield submit, root, fail
    if not logs_existed and (ROOT / "logs").is_dir() and not any((ROOT / "logs").iterdir()):
        (ROOT / "logs").rmdir()


def listed(submit, wave):
    """The cells of a wave as (name, job tag, shards, time, runner, [arguments])."""
    res, _ = submit("--wave", str(wave), "--list")
    assert res.returncode == 0, res.stderr
    out = []
    for line in res.stdout.splitlines()[1:]:
        name, tag, shards, time, runner, *args = line.split()
        out.append((name, tag, int(shards), time, runner, args))
    return out


def test_waves_have_the_planned_cells(cluster):
    submit, _, _ = cluster
    wave1 = listed(submit, 1)
    names = [c[0] for c in wave1]
    estimators = ["lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "MCP-lla", "SCAD-lla", "adaptive"]
    assert names == [f"direct_{e}_{c}" for c in ("C2I", "Cresc") for e in estimators]
    assert sum(c[2] for c in wave1) == 128                       # tasks per sample size
    assert all("--select search" in " ".join(c[5]) for c in wave1)
    assert [c[0] for c in listed(submit, 2)] == ["search_p10_C2I", "search_p10_Cresc", "search_p20_Cresc"]
    wave3 = listed(submit, 3)
    assert [c[0] for c in wave3] == [f"loglik_{e}_{c}" for c in ("C2I", "Cresc")
                                     for e in ("lasso", "lasso-up", "MCP", "MCP-up")]
    wave4 = listed(submit, 4)
    assert len(wave4) == 60 and sum(c[2] for c in wave4) == 784
    assert [c[0] for c in wave4][:6] == [f"direct_{e}_C2I_p15" for e in
                                         ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")]
    assert all("--select bic" in " ".join(c[5]) and "--select search" not in " ".join(c[5]) for c in wave4)
    assert all(c[3] == ("24:00:00" if int(c[0].rsplit("_p", 1)[1]) >= 30 else "12:00:00") for c in wave4)
    wave5 = listed(submit, 5)
    assert [c[0] for c in wave5] == [f"direct_{e}-ml_{c}" for c in ("C2I", "Cresc")
                                     for e in ("lasso", "MCP-up", "adaptive")] + \
        ["search100s_p10_Cresc", "search100u_p10_Cresc", "search30s_p20_Cresc", "search300s_p10_Cresc"] + \
        [f"direct_{e}-ebic_{c}" for c in ("C2I", "Cresc") for e in ("lasso", "MCP-up", "adaptive")] + \
        ["searche1_p10_C2I", "searche1_p10_Cresc", "searche1_p20_Cresc"]
    for name, _, _, _, runner, args in wave5:
        if runner == "run_s1_shard.py":
            assert "--select search" in " ".join(args)
            assert ("--refit loglik" in " ".join(args)) == ("-ml" in name)
            assert ("--ebic-gamma 0.5 1" in " ".join(args)) == ("-ebic" in name)
    assert sum(c[2] for c in wave5) == 196
    wave6 = listed(submit, 6)
    assert [c[0] for c in wave6] == [f"rescore1_{c[0]}" for c in wave1]
    assert [c[2] for c in wave6] == [c[2] for c in wave1]        # one task per source shard
    assert all(c[4] == "rescore_shard.py" and "--ebic-gamma 1" in " ".join(c[5]) for c in wave6)
    five = ("lasso", "MCP", "MCP-up", "MCP-up-lasso", "adaptive")
    wave7 = listed(submit, 7)                                     # (a) the log-likelihood loss over p
    assert [c[0] for c in wave7] == [x for c in ("C2I", "Cresc") for x in
                                     [f"loglik_MCP-up-lasso_{c}", f"loglik_adaptive_{c}"] +
                                     [f"loglik_{e}_{c}_p20" for e in five]]   # p = 30 left out for cost
    assert sum(c[2] for c in wave7) == 244
    assert all("--select search" in " ".join(c[5]) and "--refit" not in c[5] for c in wave7)
    wave8 = listed(submit, 8)                                     # (b) likelihood refit, (c) 100 starts
    names8 = [c[0] for c in wave8]
    assert len(wave8) == 30
    for c in ("C2I", "Cresc"):
        assert {f"loglik_{e}-ml_{c}" for e in ("lasso", "lasso-up", "MCP", "MCP-up", "MCP-up-lasso",
                                               "adaptive")} <= set(names8)
        assert {f"direct_{e}-ml_{c}" for e in ("lasso", "MCP-up", "adaptive")} <= set(names8)
        assert {f"loglik_{e}-ml_{c}_p20" for e in five} <= set(names8)
        assert f"search100sml_p10_{c}" in names8
    shards = {c[0]: c[2] for w in (1, 3, 7) for c in listed(submit, w)}
    for name, _, n_shards, _, runner, args in wave8:              # one task per source shard
        if runner == "rescore_shard.py":
            assert n_shards == shards[args[args.index("--source-cell") + 1]], name
    for wave in (1, 2, 3, 4, 5, 6, 7, 8):                         # job names: unique, short enough
        tags = [c[1] for c in listed(submit, wave)]
        assert len(set(tags)) == len(tags) and all(len(t) + 1 < 10 for t in tags)


@pytest.mark.parametrize("wave", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_cell_is_accepted_by_its_runner(cluster, wave):
    """The arguments the submit script would pass parse with the runner's own
    command line and describe an implemented combination, with the C and the
    path order that the cell's name promises."""
    submit, _, _ = cluster
    for name, _, _, _, runner, args in listed(submit, wave):
        module = {"run_s1_shard.py": run_s1_shard, "run_search_shard.py": run_search_shard,
                  "rescore_shard.py": rescore_shard}[runner]
        ns = module.build_parser().parse_args(
            ["--shard", "0", "--n-shards", "4", "--out-dir", "x", "--n-obs", "inf", *args])
        if module is rescore_shard:
            if ns.refit == "loglik":                              # wave 8: a complete cell of its own
                loss, est, cs, label = re.fullmatch(r"(direct|loglik)_([A-Za-z-]+)_(C2I|Cresc)(_p\d+)?",
                                                    ns.source_cell).groups()
                assert name == f"{loss}_{est}-ml_{cs}{label or ''}"
                assert ns.select == ("bic" if label else "search")    # the likelihood search: p = 10 only
                assert ns.p == ([10] if loss == "direct" else [])     # the direct cells hold p = 10 and 20
            else:                                                 # wave 6: an overlay
                assert name == f"rescore1_{ns.source_cell}" and ns.ebic_gamma == [1.0] and ns.select == "search"
            continue
        assert ns.c_scale == ("variance" if "_Cresc" in name else "identity")
        if module is run_search_shard:                            # waves 2, 5b, 5c and 8 (c)
            ml = name.startswith("search100sml")                 # 100 starts, likelihood refit
            restart_cell = name.startswith(("search100", "search30", "search300")) and not ml
            assert ns.methods == (["pure"] if restart_cell else ["pure", "truth"])
            assert ns.starts == ("uniform" if "100u" in name else "sparse")
            assert ns.restarts == (300 if "search300" in name else 100 if "search100" in name
                                   else 30 if "search30" in name else 10)
            assert ns.ebic_gamma == (1.0 if "searche1" in name else 0.0)
            assert ns.refit == ("loglik" if ml else "direct") and ns.add_screen == (20 if ml else None)
            assert (ns.reps, tuple(ns.p)) == ((5, (20,)) if "search30s" in name else
                                              (25, (20,)) if "p20" in name else
                                              (2, (10,)) if ml else (25, (10,)))
            continue
        assert ns.refit == ("loglik" if "-ml" in name else "direct") and ns.add_screen is None
        assert ns.ebic_gamma == ([0.5, 1.0] if "-ebic" in name else [])
        cfg = run_s1_shard.config_from_args(ns)                   # raises if not implemented
        assert cfg.loss == name.split("_")[0]
        assert (cfg.direction == "up") == ("-up" in name)
        assert (cfg.method == "lla") == ("-lla" in name)
        assert (cfg.method == "adaptive") == ("adaptive" in name)
        est = name.split("_")[1]
        assert cfg.penalty == ("lasso" if est.startswith(("lasso", "adaptive")) else est.split("-")[0])
        assert (cfg.up_start == "lasso") == ("-up-lasso" in name)
        label = re.search(r"_p(\d+)$", name)
        expected_p = ((int(label.group(1)),) if label else
                      (10, 20) if wave == 1 or "-ebic" in name else (10,))
        assert cfg.n_rep == 25 and cfg.p_values == expected_p


def test_submits_once_reports_status_and_fills_missing_shards(cluster):
    submit, root, _ = cluster
    res, calls = submit("--wave", "1", "--n", "inf", "--only", "MCP-up")
    assert res.returncode == 0, res.stderr
    assert len(calls) == 2 and "submitted 32 tasks in 2 cells" in res.stdout
    run = root / "direct_MCP-up_Cresc_ninf"
    assert (f"--array=0-15 --time=12:00:00 -J 1Muri cluster/campaign_array.sbatch "
            f"simulations/run_s1_shard.py {run} --p 10 20 --reps 25 --select search "
            f"--c-scale variance --penalty MCP --direction up --n-obs inf") in calls[1]
    assert (run / "n_shards").read_text().strip() == "16"

    res, calls = submit("--wave", "1", "--n", "inf", "--only", "MCP-up")      # already submitted
    assert len(calls) == 2 and res.stdout.count("0/16 shards written") == 2
    res, calls = submit("--wave", "1", "--n", "inf", "--status")              # never submits
    assert len(calls) == 2
    assert res.stdout.count("not started") == 14 and res.stdout.count("submitted  ") == 2

    (run / "shards").mkdir()
    for i in set(range(16)) - {3, 11}:
        (run / "shards" / f"shard_{i:04d}_of_0016.npz").touch()
    res, calls = submit("--wave", "1", "--n", "inf", "--only", "MCP-up_Cresc", "--fill",
                        "--time", "24:00:00")
    assert len(calls) == 3 and "--array=3,11" in calls[2] and "--time=24:00:00" in calls[2]
    for i in (3, 11):
        (run / "shards" / f"shard_{i:04d}_of_0016.npz").touch()
    res, calls = submit("--wave", "1", "--n", "inf", "--only", "MCP-up_Cresc", "--fill")
    assert len(calls) == 3 and "complete" in res.stdout


def test_a_refused_cell_records_nothing_and_the_rerun_continues(cluster):
    """LRZ refuses submissions beyond its per-user limit.  The script stops at the
    refused cell without recording it; the same command later submits the rest."""
    submit, root, fail = cluster
    fail.write_text("3")                                          # the third sbatch call fails
    res, calls = submit("--wave", "2", "--n", "1e4")
    assert res.returncode == 1 and len(calls) == 2
    assert "run the same command again later" in res.stderr
    assert (root / "search_p10_C2I_n1e4" / "n_shards").exists()
    assert not (root / "search_p20_Cresc_n1e4" / "n_shards").exists()
    fail.unlink()
    res, calls = submit("--wave", "2", "--n", "1e4")
    assert res.returncode == 0 and len(calls) == 3
    assert "search_p20_Cresc_n1e4" in calls[2] and "run_search_shard.py" in calls[2]
    assert "submitted 16 tasks in 1 cells" in res.stdout


def test_wave_4_runs_over_p_at_n_1000(cluster):
    """Wave 4 cells are one per p; --p selects them, n is 1000 unless --n is given, and
    the folder name carries the p and the sample size."""
    submit, root, _ = cluster
    res, calls = submit("--wave", "4", "--p", "25", "--only", "adaptive")
    assert res.returncode == 0, res.stderr
    assert len(calls) == 2 and "submitted 8 tasks in 2 cells" in res.stdout
    assert (f"-J 4adi253 cluster/campaign_array.sbatch simulations/run_s1_shard.py "
            f"{root / 'direct_adaptive_C2I_p25_n1000'} --p 25 --reps 25 --select bic "
            f"--c-scale identity --method adaptive --n-obs 1000") in calls[0]
    assert (root / "direct_adaptive_Cresc_p25_n1000" / "n_shards").read_text().strip() == "4"
    res, calls = submit("--wave", "4", "--p", "25", "--n", "inf", "--only", "adaptive_Cresc")
    assert res.returncode == 0 and len(calls) == 3
    assert f"{root / 'direct_adaptive_Cresc_p25_ninf'} --p 25" in calls[2] and "--n-obs inf" in calls[2]
    res, calls = submit("--wave", "4", "--p", "40", "--dry-run")
    assert res.returncode == 0 and len(calls) == 3                # a dry run submits nothing
    assert res.stdout.count("--time=24:00:00") == 12 and "would submit 224 tasks in 12 cells" in res.stdout
    res, _ = submit("--wave", "4", "--p", "20")
    assert res.returncode == 2 and "unknown p" in res.stderr


def test_dry_run_and_argument_errors(cluster):
    submit, root, _ = cluster
    res, calls = submit("--wave", "3", "--dry-run")
    assert res.returncode == 0 and not calls and not root.exists()
    assert res.stdout.count("sbatch --array=") == 24 and "would submit 168 tasks in 24 cells" in res.stdout
    res, calls = submit("--wave", "1", "--n", "1000", "--shards", "3", "--only", "lasso_C2I")
    assert "--array=0-2" in calls[0] and (root / "direct_lasso_C2I_n1000" / "n_shards").read_text().strip() == "3"
    for bad in (["--list"], ["--wave", "9", "--list"], ["--wave", "1", "--n", "500"],
                ["--wave", "1", "--shards", "0"], ["--wave", "1", "--frobnicate"]):
        res, _ = submit(*bad)
        assert res.returncode == 2, bad
