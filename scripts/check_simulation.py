"""Check simulations for possible errors: missing or inconsistent parameters,
non-finite values, out-of-order time, and physically implausible fields such as a
divergent velocity away from the interface. Exits with status 1 if any check
reports an error."""

import argparse
import sys
from pathlib import Path

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.bubbleml import read_bubbleml
from boiling_data.flashx.checks import Severity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="BubbleML .hdf5 files or Flash-X case directories",
    )
    parser.add_argument(
        "--start", type=int, default=None, help="first frame of BubbleML files"
    )
    parser.add_argument(
        "--stop", type=int, default=None, help="frame of BubbleML files to stop before"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="check every nth frame of BubbleML files, to bound memory",
    )
    parser.add_argument(
        "--mirror-symmetric",
        action="store_true",
        help="Flash-X runs computed only the right half of a domain symmetric "
        "about x = 0",
    )
    return parser.parse_args()


def read(path: Path, args: argparse.Namespace) -> BoilingSimulation:
    if path.is_dir():
        return BoilingSimulation.from_flashx(
            path, mirror_symmetric=args.mirror_symmetric
        )
    return read_bubbleml(path, frames=slice(args.start, args.stop, args.stride))


def main() -> None:
    args = parse_args()
    num_errors = 0
    for path in args.paths:
        issues = read(path, args).check()
        print(f"{path}: {len(issues) or 'no'} issues")
        for issue in issues:
            print(f"  {issue}")
        num_errors += sum(issue.severity == Severity.ERROR for issue in issues)
    sys.exit(1 if num_errors else 0)


if __name__ == "__main__":
    main()
