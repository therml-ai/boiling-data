import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="cases converted at once; each holds a whole case in memory",
    )
    return parser.parse_args()


def default_workers() -> int:
    # os.cpu_count is the whole node's, not the cores a Slurm job was given
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def case_directories(sim_dir: Path, prefix: str) -> list[Path]:
    cases = sorted(
        path
        for path in sim_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if not cases:
        raise SystemExit(f"no directories starting with {prefix!r} in {sim_dir}")
    return cases


def without_first_and_last_frames(simulation: BoilingSimulation) -> BoilingSimulation:
    # Flash-X writes plot files midstep, with the exception of the first and last step.
    # So all of the fields are consistent, we drop the first and last step.
    # The first plot files is written before the first step and the last after the final step,
    return BoilingSimulation(
        fields={
            name: field.time_slice(1, -1) for name, field in simulation.fields.items()
        },
        parameters=simulation.parameters,
        time=simulation.time[1:-1],
    )


def convert_case(case: Path, output_dir: Path, mirror_symmetric: bool) -> str:
    simulation = without_first_and_last_frames(
        BoilingSimulation.from_flashx(case, mirror_symmetric=mirror_symmetric)
    )
    path = simulation.to_bubbleml(output_dir / f"{case.name}.hdf5")
    temperature = simulation.field("temperature")
    return (
        f"{case.name}: {simulation.num_timesteps} frames of "
        f"{temperature.data.shape[-2]}x{temperature.data.shape[-1]} -> {path}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = case_directories(args.sim_dir, args.prefix)
    convert = partial(
        convert_case,
        output_dir=args.output_dir,
        mirror_symmetric=args.mirror_symmetric,
    )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases))) as executor:
        for summary in executor.map(convert, cases):
            print(summary, flush=True)


if __name__ == "__main__":
    main()
