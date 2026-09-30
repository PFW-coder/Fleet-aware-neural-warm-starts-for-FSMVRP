import argparse
from pathlib import Path

import torch

from fsmvrp.problem import (
    generate_fsmfd_problems,
    generate_fsmfd_v2_problems,
    generate_problems,
    save_problems,
)
from fsmvrp.runner import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible FSMVRP test set.")
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--problem-size", type=int, default=20)
    parser.add_argument("--vehicle-types", type=int, default=6)
    parser.add_argument(
        "--distribution",
        choices=("original", "fsmfd", "fsmfd_v2"),
        default="fsmfd_v2",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ensure-nondominated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(args.seed, deterministic=True)
    generator = {
        "original": generate_problems,
        "fsmfd": generate_fsmfd_problems,
        "fsmfd_v2": generate_fsmfd_v2_problems,
    }[args.distribution]
    generator_kwargs = {}
    if args.distribution == "original":
        generator_kwargs["ensure_nondominated"] = args.ensure_nondominated
    elif args.ensure_nondominated:
        raise ValueError("--ensure-nondominated is supported by original only.")
    problems = generator(
        args.instances,
        args.problem_size,
        args.vehicle_types,
        device=torch.device("cpu"),
        **generator_kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_problems(problems, str(args.output))
    print("saved {} instances to {}".format(args.instances, args.output))


if __name__ == "__main__":
    main()
