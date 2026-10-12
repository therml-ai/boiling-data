"""Randomly split BubbleML cases into train, val and test directories under an output
directory, each case keeping its path relative to the common parent of the inputs
(e.g. PoolBoiling-Saturated-FC72-2D/Twall_100.hdf5), along with its JSON sidecar if
it has one. split.json in the output directory records the seed, the fractions and
each split's cases."""

import argparse
import json
import math
import os
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

MODES = ("symlink", "copy", "move")
SPLITS = ("train", "val", "test")


def split_counts(num_cases: int, fractions: Sequence[float]) -> list[int]:
    """Cases per split for fractions of num_cases. Every split given a positive
    fraction gets at least one case, and the cases rounding leaves over go to the
    first split."""
    if len(fractions) != len(SPLITS):
        raise ValueError(
            f"need one fraction for each of {SPLITS}, not {list(fractions)}"
        )
    if any(fraction < 0 for fraction in fractions) or not math.isclose(
        sum(fractions), 1.0
    ):
        raise ValueError(
            f"fractions must be non-negative and sum to 1, not {list(fractions)}"
        )
    num_used = sum(fraction > 0 for fraction in fractions)
    if num_cases < num_used:
        raise ValueError(f"{num_cases} cases cannot fill {num_used} splits")
    counts = [
        max(1, math.floor(fraction * num_cases)) if fraction > 0 else 0
        for fraction in fractions
    ]
    # taking the smaller splits' minimum of one can overdraw the first
    counts[0] += num_cases - sum(counts)
    if fractions[0] > 0 and counts[0] < 1:
        raise ValueError(f"{num_cases} cases cannot fill {num_used} splits")
    return counts


def random_split(
    cases: Sequence[Path], fractions: Sequence[float], seed: int
) -> dict[str, list[Path]]:
    """cases shuffled with seed and divided into the train, val and test splits.
    The order the cases are given in does not change the split."""
    shuffled = sorted(cases)
    random.Random(seed).shuffle(shuffled)
    split: dict[str, list[Path]] = {}
    start = 0
    for name, count in zip(SPLITS, split_counts(len(shuffled), fractions), strict=True):
        split[name] = sorted(shuffled[start : start + count])
        start += count
    return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="BubbleML .hdf5 files to split"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="where to make the splits"
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="fraction of the cases in each split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="symlink",
        help="how cases enter the splits; symlinks take no space and leave the "
        "originals in place",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the split without making it"
    )
    return parser.parse_args()


def case_files(case: Path) -> list[Path]:
    """The case and the JSON sidecar older BubbleML files keep their parameters in."""
    sidecar = case.with_suffix(".json")
    return [case, sidecar] if sidecar.exists() else [case]


def place(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{destination} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def main() -> None:
    args = parse_args()
    # absolute, not resolved: a symlinked case keeps its own name and directory
    cases = [path.absolute() for path in args.paths]
    missing = [str(case) for case in cases if not case.is_file()]
    if missing:
        raise SystemExit(f"no such files: {missing}")
    root = Path(os.path.commonpath([case.parent for case in cases]))
    split = random_split(cases, args.fractions, args.seed)
    manifest: dict[str, object] = {"seed": args.seed, "fractions": args.fractions}
    for name in SPLITS:
        relative = [case.relative_to(root) for case in split[name]]
        manifest[name] = [str(path) for path in relative]
        print(f"{name}: {len(relative)} cases")
        for path in relative:
            print(f"  {path}")
            if not args.dry_run:
                for source in case_files(root / path):
                    place(
                        source,
                        args.output_dir / name / source.relative_to(root),
                        args.mode,
                    )
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "split.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=4)
            handle.write("\n")


if __name__ == "__main__":
    main()
