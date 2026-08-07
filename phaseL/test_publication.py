#!/usr/bin/env python3
"""CPU guards for the publication-scale cache indexing scheme."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phaseL.build_selection import (
    RECORD_STRIDE,
    SEED_STRIDE,
    copied_record,
    expanded_index,
    expanded_seed,
    select_stratified,
)


def main() -> None:
    source = {
        "idx": 17,
        "category": "color",
        "prompt": "a red cube and a blue sphere",
        "seed_base": 17000,
        "endpoint_vqa": [0.1, 0.5, 0.2, 0.3],
        "oracle_idx": 1,
        "random_idx": 2,
    }
    copied = copied_record(source)
    assert copied["idx"] == 17
    assert copied["source_idx"] == 17
    assert copied["seed_repeat"] == 0
    assert copied["label_source"] == "correct_prompt"
    indices = {expanded_index(17, repeat) for repeat in range(4)}
    seeds = {expanded_seed(17000, repeat) for repeat in range(4)}
    assert len(indices) == 4 and len(seeds) == 4
    assert max(indices) < 4 * RECORD_STRIDE
    assert sorted(seeds)[1] - sorted(seeds)[0] == SEED_STRIDE

    records = [
        {**source, "idx": idx, "category": category}
        for idx, category in enumerate(["color", "color", "shape", "shape"])
    ]
    selected = select_stratified(records, 1)
    assert [record["category"] for record in selected] == ["color", "shape"]
    print({"pass": True, "record_stride": RECORD_STRIDE, "seed_stride": SEED_STRIDE})


if __name__ == "__main__":
    main()
