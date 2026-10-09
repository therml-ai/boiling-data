"""Normalization statistics of BubbleML files, written as one JSON file: the count,
mean, standard deviation, min and max of each field over every frame and grid point
under "fields", and of every numeric config parameter over the files, nested by the
config's own groups and key names, under "config"."""

import argparse
import json
from pathlib import Path
from typing import Any

from boiling_data.statistics import dataset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="BubbleML .hdf5 files to include"
    )
    parser.add_argument(
        "--fields", nargs="+", required=True, help="fields to compute statistics of"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write")
    parser.add_argument(
        "--frames-per-read",
        type=int,
        default=64,
        help="frames of one field held in memory at a time",
    )
    return parser.parse_args()


def write_json(path: Path, record: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=4)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        dataset_statistics(args.paths, args.fields, args.frames_per_read),
    )
    print(f"statistics of {len(args.paths)} files -> {args.output}")


if __name__ == "__main__":
    main()
