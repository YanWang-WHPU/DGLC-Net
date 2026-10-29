"""Run a checked-in CLI recipe without a shell; use --dry-run to inspect it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "train_hsi_uda.py",
    "eval_ensemble_proto_refine.py",
    "evaluate_hsi_ensemble.py",
    "apply_local_consensus_router.py",
    "evaluate_predictions.py",
}


def build_command(config: dict) -> list[str]:
    script = config["script"]
    if script not in SCRIPTS:
        raise ValueError(f"Unsupported script: {script}")
    args = config["args"]
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("Config args must be a list of strings")
    return [sys.executable, str(ROOT / "scripts" / script), *args]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = build_command(config)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
