"""Run every test suite (each in its own process) and summarise.

Usage:  python tests/run_all.py [name ...]      e.g.  python tests/run_all.py profile sections

Each suite prints PASS / FAIL lines; GUI suites open a ScanProfiler window briefly.
Exit code is non-zero if any check failed or a suite crashed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ORDER = [  # fast core suites first, GUI suites after
    "profile", "measure", "fitting", "alignment", "sections", "exports",
    "selection", "visibility", "fit_e2e", "align_e2e", "sections_e2e", "export_e2e", "module5",
]


def main(argv: list[str]) -> int:
    wanted = argv or ORDER
    sys.path.insert(0, str(TESTS))
    import _common  # noqa: F401, E402  generate test parts once, before the suites run

    _common.ensure_all()
    total_pass = total_fail = 0
    broken = []
    for name in wanted:
        script = TESTS / f"suite_{name}.py"
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-u", str(script)], capture_output=True, text=True, cwd=TESTS.parent)
        out = proc.stdout + proc.stderr
        passes = len(re.findall(r"(?m)^PASS", out))
        fails = re.findall(r"(?m)^FAIL.*$", out)
        crashed = proc.returncode != 0 or "ERRORS:" in out or "Traceback" in proc.stdout
        total_pass += passes
        total_fail += len(fails)
        status = "ok" if not fails and not crashed else "FAILED"
        print(f"{status:7s} {name:14s} {passes:3d} passed  {len(fails):2d} failed  ({time.perf_counter() - t0:5.1f}s)")
        for line in fails:
            print("        " + line)
        if crashed:
            broken.append(name)
            print("        " + "\n        ".join(out.strip().splitlines()[-8:]))
    print(f"\n{total_pass} checks passed, {total_fail} failed" + (f", crashed: {', '.join(broken)}" if broken else ""))
    return 1 if total_fail or broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
