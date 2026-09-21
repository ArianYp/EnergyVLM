"""Where the evaluation RECORDS live.

The analysis scripts under eval/ (and paper/verify_numbers.py) recompute their tables from per-prompt
score dumps -- phaseN/eval_<label>_<job>/ and phaseN/eval10_<label>_<job>/ trees written by
scripts/eval_alignment.lsf -- which are not shipped with this repository. They are found under an
artifact root, resolved in this order:

    --artifacts <dir>          on the command line
    $ENERGYVLM_ARTIFACTS       in the environment
    the current directory      if it holds phaseN/
    the repository root        if it holds phaseN/

The scripts chdir there and glob relative paths, so the layout under the root is the experimental
tree's (phaseN/, phaseT/, phaseW/). Files that belong to THIS repository (docs/, paper/) are always
resolved against the repository root, so the two roots can differ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifact_root(explicit: str | None = None, need: str = "phaseN") -> Path:
    """Resolve the artifact root without changing directory; raises SystemExit if none holds `need`."""
    cands = [explicit, os.environ.get("ENERGYVLM_ARTIFACTS"), os.getcwd(), str(ROOT)]
    for c in cands:
        if c and (Path(c) / need).is_dir():
            return Path(c).resolve()
    raise SystemExit(f"[artifacts] no evaluation records found (looked for a {need}/ directory under "
                     f"{[c for c in cands if c]}). They are produced by scripts/eval_alignment.lsf; point "
                     f"--artifacts or $ENERGYVLM_ARTIFACTS at the tree that holds them.")


def chdir_artifacts(explicit: str | None = None, need: str = "phaseN") -> Path:
    """artifact_root(), then chdir into it (the scripts glob relative paths)."""
    root = artifact_root(explicit, need)
    os.chdir(root)
    return root


def add_artifacts_arg(parser) -> None:
    parser.add_argument("--artifacts", default=None,
                        help="root holding the phaseN/ evaluation records (default: $ENERGYVLM_ARTIFACTS, then cwd)")


def strip_artifacts_arg(argv: list[str] | None = None) -> tuple[str | None, list[str]]:
    """For scripts that do not use argparse: pull `--artifacts <dir>` out of argv."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--artifacts" in argv:
        i = argv.index("--artifacts")
        val = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
        return val, argv
    return None, argv
