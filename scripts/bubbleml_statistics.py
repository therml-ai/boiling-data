"""Normalization statistics of BubbleML files: the global mean and standard deviation
of each field and the min / max of the non-dimensional and heater parameters."""

import argparse
from pathlib import Path

import yaml

from boiling_data.statistics import (
    bubbleml_parameter_ranges,
    field_moments,
    statistics_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="BubbleML .hdf5 files to include"
    )
    parser.add_argument(
        "--fields", nargs="+", required=True, help="fields to compute statistics of"
    )
    parser.add_argument("--output", type=Path, required=True, help="yaml file to write")
    parser.add_argument(
        "--frames-per-read",
        type=int,
        default=64,
        help="frames of one field held in memory at a time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = statistics_record(
        field_moments(args.paths, args.fields, args.frames_per_read),
        bubbleml_parameter_ranges(args.paths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)
    print(f"statistics of {len(args.paths)} files -> {args.output}")


if __name__ == "__main__":
    main()
