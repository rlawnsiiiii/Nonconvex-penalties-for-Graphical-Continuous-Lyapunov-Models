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
           "CAMPAIGN_ROOT": str(root), "SUBMIT_LOG": str(tmp_path / "submissions.log")}
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
    for wave in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):                  # job names: unique, short enough
        tags = [c[1] for c in listed(submit, wave)]
        assert len(set(tags)) == len(tags) and all(len(t) + 1 < 10 for t in tags)


STARTS9 = ("exact", "l1", "l2", "truth0.05", "truth0.2")
FOUR = ("lasso", "MCP", "MCP-up", "adaptive")
THREE = ("lasso", "MCP-up", "adaptive")          # 19 Oct: MCP sparse -> dense dropped from the knob groups


def test_wave_9_has_the_planned_groups(cluster):
    """Plan of 13 October 2026: E1 starts (20 cells per n), E2 threshold (4), E3 betamin (24 since MCP sparse
    -> dense was dropped on 19 October), E4 class (6: dag only), E5 twocycle (30) and the optional loglik starts (20);
    --group selects one; the loglik starts run at n = 1000 and 1e4 only, the knob groups also at inf."""
    submit, root, _ = cluster
    wave9 = listed(submit, 9)
    names = [c[0] for c in wave9]
    # shards: starts 2 C x 5 starts x (16 + 4); threshold 4 x 4; betamin 4 w x 2 C x (4 + 16 + 4); class 1 g x 2 C x 24;
    # twocycle 5 s x 2 C x (4 + 16 + 4); loglik starts 5 x 2 x (8 + 16)
    assert len(wave9) == 104 and sum(c[2] for c in wave9) == 200 + 16 + 192 + 48 + 240 + 240
    groups = {"starts": [f"direct_{pr}-{st}_{c}" for c in ("C2I", "Cresc") for st in STARTS9 for pr in ("MCP-up", "adaptive")],
              "threshold": [f"direct_threshold{v}_{c}" for c in ("C2I", "Cresc") for v in ("", "-norefit")],
              "betamin": [f"direct_{e}_{c}_w{w}" for w in ("0", "01", "025", "05") for c in ("C2I", "Cresc") for e in THREE],
              "class": [f"direct_{e}_{c}_{g}" for g in ("dag",) for c in ("C2I", "Cresc") for e in THREE],
              "twocycle": [f"direct_{e}_{c}_s{sh}" for sh in ("0", "025", "05", "075", "1") for c in ("C2I", "Cresc")
                           for e in ("lasso", "MCP-up", "adaptive")],
              "starts-loglik": [f"loglik_{pr}-{st}_{c}" for c in ("C2I", "Cresc") for st in STARTS9 for pr in ("MCP-up", "adaptive")]}
    assert names == sum(groups.values(), [])
    for group, cells in groups.items():
        res, _ = submit("--wave", "9", "--group", group, "--list")
        assert [line.split()[0] for line in res.stdout.splitlines()[1:]] == cells, group
    for name, _, shards, time, runner, args in wave9:
        a = " ".join(args)
        assert runner == "run_s1_shard.py" and "--select search" in a
        if name.startswith("loglik"):
            assert "--loss loglik --p 10 " in a and time == "24:00:00" and shards == (8 if "C2I" in name else 16)
        else:
            assert "--p 10 20 " in a and time == "12:00:00" and "--loss" not in a
        est = name.split("_")[1]
        if "-" in est and est.split("-", 1)[0] in ("MCP", "adaptive") and est not in ("MCP-up",):
            start = est.rsplit("-", 1)[1].replace("truth", "truth-")
            assert (f"--up-start {start}" in a) == est.startswith("MCP-up") and (f"--pilot {start}" in a) == est.startswith("adaptive")
        if "_w" in name:
            w = name.rsplit("_w", 1)[1]
            assert f"--weights uniform --w-min {w[0]}{'.' + w[1:] if len(w) > 1 else ''}" in a
            assert "--ebic-gamma 0.5 1" in a and "--c-choices C_ID" in a      # 19 Oct: every reduction, true C = 2I only
        elif re.search(r"_s\d+$", name):                                       # E5: the 2-cycle share (19 Oct)
            sh = name.rsplit("_s", 1)[1]
            assert f"--graph pairs --two-cycle-share {sh[0]}{'.' + sh[1:] if len(sh) > 1 else ''}" in a
            assert "--ebic-gamma 0.5 1" in a and "--c-choices C_ID" in a and "--weights" not in a
        elif name.endswith("_dag"):                                            # E4: every reduction, all four true C
            assert f"--graph {name.rsplit('_', 1)[1]}" in a
            assert "--ebic-gamma 0.5 1" in a and "--c-choices" not in a
        else:
            assert "--ebic-gamma" not in a and "--c-choices" not in a         # the other groups are as planned
        assert "_MCP_" not in name                                            # 19 Oct: MCP sparse -> dense dropped
    # the loglik starts are not planned at n = inf: reported, nothing submitted; the three knob groups are (19 October)
    res, calls = submit("--wave", "9", "--group", "starts-loglik", "--n", "inf", "--dry-run")
    assert res.returncode == 0 and not calls and res.stdout.count("not planned") == 20
    assert "would submit 0 tasks in 0 cells" in res.stdout
    res, calls = submit("--wave", "9", "--group", "betamin", "--n", "inf", "--dry-run")
    assert res.returncode == 0 and res.stdout.count("sbatch --array=") == 24 and "would submit 192 tasks in 24 cells" in res.stdout
    assert "--n-obs inf" in res.stdout and "--c-choices C_ID" in res.stdout and "not planned" not in res.stdout
    res, calls = submit("--wave", "9", "--group", "twocycle", "--n", "inf", "--dry-run")
    assert res.returncode == 0 and res.stdout.count("sbatch --array=") == 30 and "would submit 240 tasks in 30 cells" in res.stdout
    res, calls = submit("--wave", "9", "--group", "class", "--n", "inf", "--dry-run")
    assert res.returncode == 0 and res.stdout.count("sbatch --array=") == 6 and "would submit 48 tasks in 6 cells" in res.stdout
    res, calls = submit("--wave", "9", "--group", "class", "--n", "1000", "--dry-run")
    assert res.returncode == 0 and res.stdout.count("sbatch --array=") == 6 and "would submit 48 tasks in 6 cells" in res.stdout
    res, calls = submit("--wave", "9", "--group", "starts", "--n", "inf", "--only", "adaptive-l1", "--dry-run")
    assert res.returncode == 0 and res.stdout.count("sbatch --array=") == 2
    assert f"{root / 'direct_adaptive-l1_Cresc_ninf'} --p 10 20 --reps 25 --select search --c-scale variance --method adaptive --pilot l1 --n-obs inf" in res.stdout
    for bad in (["--wave", "9", "--group", "bogus", "--list"], ["--wave", "1", "--group", "starts", "--list"]):
        res, _ = submit(*bad)
        assert res.returncode == 2, bad
    # a trailing $ anchors an --only pattern at the end of the name: three weight levels without w_min = 0.1
    res, _ = submit("--wave", "9", "--group", "betamin", "--only", "_w0$", "_w025", "_w05", "--list")
    names = [line.split()[0] for line in res.stdout.splitlines()[1:]]
    assert len(names) == 18 and not any(n.endswith("_w01") for n in names) and sum(n.endswith("_w0") for n in names) == 6
    res, _ = submit("--wave", "9", "--group", "betamin", "--only", "_w0", "--list")
    assert len(res.stdout.splitlines()) - 1 == 24                   # without the anchor every level matches


@pytest.mark.parametrize("wave", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
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
            assert ns.start_blocks == (4 if ml else 1)
            assert (ns.reps, tuple(ns.p)) == ((5, (20,)) if "search30s" in name else
                                              (25, (20,)) if "p20" in name else
                                              (2, (10,)) if ml else (25, (10,)))
            continue
        assert (ns.refit or "direct") == ("loglik" if "-ml" in name else "direct") and ns.add_screen is None
        knob = re.search(r"_(w|s)\d+$", name) is not None                # betamin / twocycle: true C = 2I only (19 Oct)
        pd = "-pd_" in name                                                # wave 10: the diagonal penalised, C_ID
        every = knob or name.endswith("_dag") or pd                        # every reduction
        assert ns.ebic_gamma == ([0.5, 1.0] if ("-ebic" in name or every) else [])
        assert ns.c_choices == (["C_ID"] if knob or pd else None)
        cfg = run_s1_shard.config_from_args(ns)                   # raises if not implemented
        assert cfg.penalize_diagonal == pd
        assert cfg.loss == name.split("_")[0]
        assert (cfg.direction == "up") == ("-up" in name)
        assert (cfg.method == "lla") == ("-lla" in name)
        assert (cfg.method == "adaptive") == ("adaptive" in name)
        est = name.split("_")[1]
        assert cfg.penalty == ("lasso" if est.startswith(("lasso", "adaptive", "threshold")) else est.split("-")[0])
        assert (cfg.up_start == "lasso") == ("-up-lasso" in name)
        label = re.search(r"_p(\d+)$", name)
        expected_p = ((int(label.group(1)),) if label else
                      (10, 20) if wave in (1, 10) or "-ebic" in name or (wave == 9 and cfg.loss == "direct") else (10,))
        assert cfg.n_rep == 25 and cfg.p_values == expected_p
        if wave == 9:                                             # the start, the pilot and the generator knobs
            start = est.rsplit("-", 1)[1].replace("truth", "truth-") if est.startswith(("MCP-up-", "adaptive-")) else "lasso"
            assert cfg.start_name == (start if est.startswith("MCP-up-") else "lasso" if cfg.loss == "direct" else "exact")
            assert cfg.pilot == (start if est.startswith("adaptive-") else "lasso")
            assert cfg.method == ("threshold" if est == "threshold" else "threshold-norefit" if est == "threshold-norefit"
                                  else "adaptive" if est.startswith("adaptive") else "path")
            m = re.search(r"_w([0-9]+)_|_(dag)_|_s([0-9]+)_", name + "_")
            decimal = lambda digits: float(f"{digits[0]}.{digits[1:]}") if len(digits) > 1 else float(digits)  # noqa: E731
            assert cfg.weights == ("uniform" if m and m.group(1) else "normal")
            assert cfg.w_min == (decimal(m.group(1)) if m and m.group(1) else 0.0)
            assert cfg.graph == (m.group(2) if m and m.group(2) else "pairs" if m and m.group(3) else "random")
            assert cfg.two_cycle_share == (decimal(m.group(3)) if m and m.group(3) else 0.0)   # E5 (19 Oct)


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


def test_wave_10_reruns_the_briefing_with_the_diagonal_penalised(cluster):
    """21 October 2026: the direct-loss cells of the briefing (waves 1 and 4 without LLA) with
    --penalize-diagonal on the true C = 2I graphs; p = 10 and 20 at every n with the search, p = 15 ... 50
    at n = 1000 only with the selection (--p as in wave 4)."""
    submit, root, _ = cluster
    six = ("lasso", "MCP", "SCAD", "MCP-up", "SCAD-up", "adaptive")
    wave10 = listed(submit, 10)
    names = [c[0] for c in wave10]
    assert names[:12] == [f"direct_{e}-pd_{c}" for c in ("C2I", "Cresc") for e in six]
    assert names[12:] == [f"direct_{e}-pd_{c}_p{p}" for p in (15, 25, 30, 40, 50) for c in ("C2I", "Cresc") for e in six]
    for name, _, _, time, _, args in wave10:
        a = " ".join(args)
        assert "--penalize-diagonal" in a and "--c-choices C_ID" in a and "--ebic-gamma 0.5 1" in a
        assert ("--select bic" in a) == ("_p" in name) and ("--select search" in a) == ("_p" not in name)
        assert time == ("24:00:00" if re.search(r"_p(40|50)$", name) else "12:00:00")
    res, calls = submit("--wave", "10", "--n", "1e4", "--dry-run")
    assert res.returncode == 0 and not calls and not root.exists()
    assert "would submit 32 tasks in 12 cells" in res.stdout and res.stdout.count("not planned") == 60
    res, _ = submit("--wave", "10", "--dry-run")
    assert "would submit 276 tasks in 96 cells" in res.stdout
    res, calls = submit("--wave", "10", "--n", "1000", "--p", "40", "--only", "adaptive-pd_Cresc")
    assert res.returncode == 0 and len(calls) == 2
    assert f"-J 10adr3 cluster/campaign_array.sbatch simulations/run_s1_shard.py {root / 'direct_adaptive-pd_Cresc_n1000'}" in calls[0]
    assert f"{root / 'direct_adaptive-pd_Cresc_p40_n1000'} --p 40 --select bic" in calls[1]
    res, _ = submit("--wave", "10", "--p", "20", "--list")
    assert res.returncode == 2 and "unknown p for wave 10" in res.stderr


def test_dry_run_and_argument_errors(cluster):
    submit, root, _ = cluster
    res, calls = submit("--wave", "3", "--dry-run")
    assert res.returncode == 0 and not calls and not root.exists()
    assert res.stdout.count("sbatch --array=") == 24 and "would submit 168 tasks in 24 cells" in res.stdout
    res, calls = submit("--wave", "1", "--n", "1000", "--shards", "3", "--only", "lasso_C2I")
    assert "--array=0-2" in calls[0] and (root / "direct_lasso_C2I_n1000" / "n_shards").read_text().strip() == "3"
    for bad in (["--list"], ["--wave", "11", "--list"], ["--wave", "1", "--n", "500"],
                ["--wave", "1", "--shards", "0"], ["--wave", "1", "--frobnicate"]):
        res, _ = submit(*bad)
        assert res.returncode == 2, bad


def test_fill_uses_the_cells_record_and_refuses_another_waves(cluster):
    """Code review C1 / C3 (13 October 2026): a cell folder records what it was submitted with
    (wave, runner, arguments, time), --fill resubmits that, and a cell recorded by another wave
    -- six names belong to wave 5 and wave 8 with different runners and shard counts -- is
    reported and skipped instead of being filled with the wrong runner.  A folder without a
    record is filled only when its shard count matches the wave's definition.  Every sbatch
    call is appended to the submissions log with the commit."""
    submit, root, _ = cluster
    res, calls = submit("--wave", "5", "--n", "1e4", "--only", "lasso-ml_C2I")
    assert res.returncode == 0, res.stderr
    assert len(calls) == 1
    run = root / "direct_lasso-ml_C2I_n1e4"
    record = (run / "submission").read_text().strip().split("|")
    assert record[:2] == ["5", "simulations/run_s1_shard.py"] and "--refit loglik" in record[2] and record[3] == "12:00:00"
    (run / "shards").mkdir()
    for i in range(7):
        (run / "shards" / f"shard_{i:04d}_of_0008.npz").touch()
    res, calls = submit("--wave", "8", "--n", "1e4", "--only", "lasso-ml_C2I", "--fill")
    assert res.returncode == 0 and len(calls) == 1
    assert "other wave" in res.stdout and "--wave 5 --fill" in res.stdout
    res, calls = submit("--wave", "5", "--n", "1e4", "--only", "lasso-ml_C2I", "--fill")
    assert res.returncode == 0 and len(calls) == 2
    assert "--array=7" in calls[1] and "run_s1_shard.py" in calls[1] and "--refit loglik" in calls[1]
    # a folder from before the record existed: filled only if the shard count matches the wave
    legacy = root / "direct_MCP-up-ml_Cresc_n1e4"
    (legacy / "shards").mkdir(parents=True)
    (legacy / "n_shards").write_text("8\n")
    res, calls = submit("--wave", "8", "--n", "1e4", "--only", "MCP-up-ml_Cresc", "--fill")   # wave 8: 16 shards
    assert res.returncode == 0 and len(calls) == 2 and "mismatch" in res.stdout
    res, calls = submit("--wave", "5", "--n", "1e4", "--only", "MCP-up-ml_Cresc", "--fill")   # wave 5: 8 shards
    assert res.returncode == 0 and len(calls) == 3 and "--array=0,1,2,3,4,5,6,7" in calls[2]
    log = (root.parent / "submissions.log").read_text().splitlines()
    assert len(log) == 3 and all(line.count(" | ") == 2 for line in log)
    # --only matching no cell is an error, not "nothing to do"
    res, calls = submit("--wave", "1", "--only", "no-such-cell")
    assert res.returncode == 2 and "matches --only" in res.stderr and len(calls) == 3
