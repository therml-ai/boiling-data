import argparse
from pathlib import Path

from boiling_data.boiling_data import BoilingSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help="directory holding the case directories",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="where to write <case>.hdf5"
    )
    parser.add_argument(
        "--prefix",
        default="case",
        help="only convert case directories starting with this",
    )
    parser.add_argument(
        "--mirror-symmetric",
        action="store_true",
        help="the runs are symmetric about x = 0 and only computed the right half; "
        "write the full domain",
    )
    return parser.parse_args()


def case_directories(sim_dir: Path, prefix: str) -> list[Path]:
    cases = sorted(
        path
        for path in sim_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if not cases:
        raise SystemExit(f"no directories starting with {prefix!r} in {sim_dir}")
    return cases


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in case_directories(args.sim_dir, args.prefix):
        simulation = BoilingSimulation.from_flashx(
            case, mirror_symmetric=args.mirror_symmetric
        )
        path = simulation.to_bubbleml(args.output_dir / f"{case.name}.hdf5")
        temperature = simulation.field("temperature")
        print(
            f"{case.name}: {simulation.num_timesteps} frames of "
            f"{temperature.data.shape[-2]}x{temperature.data.shape[-1]} -> {path}"
        )


if __name__ == "__main__":
    main()
