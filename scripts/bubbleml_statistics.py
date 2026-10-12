"""Normalization statistics of BubbleML files, written as one JSON file: the count,
mean, standard deviation, min and max of each field over every frame and grid point
under "fields", and of every numeric config parameter over the files, nested by the
config's own groups and key names, under "config".

The files are either given directly, or are the train cases of a split directory made
by scripts/split_dataset.py, whose statistics.json they are written to by default.
Without --fields, the statistics cover every field all the files store."""

import argparse
import json
from pathlib import Path
from typing import Any

from boiling_data.bubbleml import read_bubbleml_field_names
from boiling_data.statistics import dataset_statistics

SPLIT_MANIFEST = "split.json"
SPLIT_STATISTICS = "statistics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="*", help="BubbleML .hdf5 files to include"
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="a split directory whose train cases to include, in place of paths",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="fields to compute statistics of; every field all the files store by "
        "default",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON file to write; with --split-dir, its statistics.json by default",
    )
    parser.add_argument(
        "--frames-per-read",
        type=int,
        default=64,
        help="frames of one field held in memory at a time",
    )
    args = parser.parse_args()
    if bool(args.paths) == (args.split_dir is not None):
        parser.error("give either paths or --split-dir")
    if args.output is None:
        if args.split_dir is None:
            parser.error("--output is required with paths")
        args.output = args.split_dir / SPLIT_STATISTICS
    return args


def train_cases(split_dir: Path) -> list[Path]:
    """The train cases split.json lists, under split_dir/train."""
    with open(split_dir / SPLIT_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases = [split_dir / "train" / path for path in manifest["train"]]
    missing = [str(case) for case in cases if not case.exists()]
    if missing:
        raise FileNotFoundError(f"{split_dir} is missing train cases: {missing}")
    return cases


def common_field_names(paths: list[Path]) -> list[str]:
    """The fields every file stores, in the first file's order. The fields only some
    files store are reported and left out, as a dataset cannot be normalized by
    statistics of fields some of its files lack."""
    names = [read_bubbleml_field_names(path) for path in paths]
    common = [name for name in names[0] if all(name in other for other in names)]
    partial = sorted(
        {name for file_names in names for name in file_names} - set(common)
    )
    if partial:
        print(f"skipping fields not every file stores: {partial}")
    return common


def write_json(path: Path, record: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=4)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    paths = args.paths if args.split_dir is None else train_cases(args.split_dir)
    fields = args.fields or common_field_names(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        dataset_statistics(paths, fields, args.frames_per_read),
    )
    print(f"statistics of {fields} over {len(paths)} files -> {args.output}")


if __name__ == "__main__":
    main()
