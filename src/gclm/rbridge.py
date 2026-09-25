"""Bridge to the R solver backends.

Used by :func:`gclm.lasso.lasso_path` when ``solver`` is ``"glmnet"`` or
``"ncvreg"``, and by the validation tests.  Communication is a JSON file pair
over a subprocess -- no rpy2 dependency, and R stays entirely optional: nothing
here is imported unless an R backend is actually selected.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

R_DIR = Path(__file__).resolve().parents[2] / "R"


class RNotAvailable(RuntimeError):
    """Raised when an R backend is requested but R or a package is missing."""


@lru_cache(maxsize=None)
def r_available(*packages: str) -> bool:
    """True if ``Rscript`` exists and every named package can be loaded."""
    if shutil.which("Rscript") is None:
        return False
    if not packages:
        packages = ("jsonlite",)
    checks = " && ".join(
        f'requireNamespace("{pkg}", quietly=TRUE)' for pkg in packages
    )
    probe = subprocess.run(
        ["Rscript", "-e", f"q(status = if ({checks}) 0 else 1)"],
        capture_output=True,
    )
    return probe.returncode == 0


def run_r(script: str, payload: dict, packages: tuple[str, ...] = ()) -> dict:
    """Run ``R/<script>`` on ``payload``; return the parsed JSON result."""
    if packages and not r_available(*packages):
        raise RNotAvailable(
            f"{script} needs Rscript with {', '.join(packages)}. "
            f'Install with: install.packages(c({", ".join(chr(34) + p + chr(34) for p in packages)}))'
        )
    path = R_DIR / script
    if not path.exists():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        inp.write_text(json.dumps(payload))
        proc = subprocess.run(
            ["Rscript", str(path), str(inp), str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(out.read_text())
