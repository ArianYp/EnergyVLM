#!/usr/bin/env python3
"""Read the scalar history out of a local `.wandb` run file, without network access.

Task E requires recovering the already-logged preference-logit, per-branch error and gradient
distributions from the completed Phase-I runs before choosing a beta sweep. Those runs are finished,
so the numbers exist only in W&B; the local datastore file is the offline copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_history(path: Path) -> list[dict]:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    store = datastore.DataStore()
    store.open_for_scan(str(path))
    rows = []
    while True:
        blob = store.scan_data()
        if blob is None:
            break
        record = pb.Record()
        record.ParseFromString(blob)
        if record.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                value = json.loads(item.value_json)
            except json.JSONDecodeError:
                continue
            # Media and other structured payloads are not scalars; keep the history numeric.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
        if row:
            rows.append(row)
    return rows


def find_run_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("*.wandb"))
    if not candidates:
        raise SystemExit(f"no .wandb datastore under {run_dir}")
    # Prefer the resolved timestamped run over the `latest-run` symlink to avoid double counting.
    for candidate in candidates:
        if "latest-run" not in candidate.parts:
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="directory containing wandb/")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = find_run_file(Path(args.run_dir))
    rows = read_history(path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    keys = sorted({k for row in rows for k in row})
    print(json.dumps({"source": str(path), "rows": len(rows), "keys": keys}, indent=2))


if __name__ == "__main__":
    main()
