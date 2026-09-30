"""Neural FSMVRP construction followed by a bounded PyVRP improvement run."""

import argparse
import csv
import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path

import torch

from baselines.run_pyvrp import solve_instance
from fsmvrp.problem import load_problems
from fsmvrp.runner import evaluate_solution_plans, load_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FSMVRP hybrid solver.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--method-name", default="fsmvrp_neural_pyvrp")
    parser.add_argument("--rollout-size", type=int, default=100)
    parser.add_argument("--decode-type", choices=("greedy", "sampling"), default="sampling")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--augmentation-chunk-size",
        type=int,
        default=8,
        help=(
            "Number of geometric augmentations encoded simultaneously. "
            "Use a smaller value to reduce peak memory on large instances."
        ),
    )
    parser.add_argument("--prune-dominated-types", action="store_true")
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--max-runtime",
        type=float,
        help="PyVRP search time after neural construction.",
    )
    time_group.add_argument(
        "--max-total-runtime",
        type=float,
        help="Per-instance wall-time budget including neural construction and PyVRP.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--objective-precision", type=int, default=10000)
    parser.add_argument("--load-precision", type=int, default=1000000000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_limit = (
        args.max_runtime
        if args.max_runtime is not None
        else args.max_total_runtime
    )
    if requested_limit <= 0:
        raise ValueError("The selected runtime limit must be positive.")
    if args.rollout_size <= 0:
        raise ValueError("rollout-size must be positive.")
    if not 1 <= args.augmentation_chunk_size <= 8:
        raise ValueError("augmentation-chunk-size must be between 1 and 8.")
    if args.objective_precision <= 0 or args.load_precision <= 0:
        raise ValueError("Precision factors must be positive.")

    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)
    model, checkpoint = load_model(args.checkpoint, device)
    loaded = load_problems(str(args.data), device)
    dataset_name = args.dataset_name or args.data.stem

    rows = []
    for index in range(loaded.batch_size):
        instance = loaded.slice(index, index + 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        total_start = time.perf_counter()
        neural_start = time.perf_counter()
        plans = evaluate_solution_plans(
            model,
            instance,
            rollout_size=args.rollout_size,
            decode_type=args.decode_type,
            augment=args.augment,
            prune_dominated_types=args.prune_dominated_types,
            augmentation_chunk_size=args.augmentation_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_gpu_memory_mb = 0.0
        neural_seconds = time.perf_counter() - neural_start

        remaining_wall_runtime = None
        if args.max_total_runtime is not None:
            elapsed_before_pyvrp = time.perf_counter() - total_start
            remaining_wall_runtime = args.max_total_runtime - elapsed_before_pyvrp
            if remaining_wall_runtime <= 0:
                raise RuntimeError(
                    "Neural construction used {:.6f}s, exceeding the {:.6f}s "
                    "total wall-time budget.".format(
                        elapsed_before_pyvrp, args.max_total_runtime
                    )
                )

        plan = plans[0]
        cpu_instance = instance.to(torch.device("cpu"))
        outcome = solve_instance(
            cpu_instance,
            0,
            max_runtime=args.max_runtime,
            seed=args.seed,
            objective_precision=args.objective_precision,
            load_precision=args.load_precision,
            initial_routes=plan.routes,
            max_wall_runtime=remaining_wall_runtime,
        )
        total_seconds = time.perf_counter() - total_start
        improvement_pct = 100.0 * (plan.cost - outcome["cost"]) / plan.cost
        fleet_by_type = Counter(route.vehicle_type for route in plan.routes)
        route_payload = [
            {"vehicle_type": route.vehicle_type, "visits": list(route.visits)}
            for route in plan.routes
        ]
        rows.append(
            {
                "dataset": dataset_name,
                "instance_id": index,
                "method": args.method_name,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch", ""),
                "training_seed": checkpoint.get("run_config", {}).get("seed", ""),
                "evaluation_seed": args.seed,
                "problem_size": instance.problem_size,
                "vehicle_types": instance.vehicle_type_num,
                "rollout_size": args.rollout_size,
                "decode_type": args.decode_type,
                "augment": int(args.augment),
                "augmentation_chunk_size": (
                    args.augmentation_chunk_size if args.augment else 1
                ),
                "prune_dominated_types": int(args.prune_dominated_types),
                "time_limit_mode": (
                    "search" if args.max_runtime is not None else "total_wall"
                ),
                "configured_search_limit_seconds": (
                    "" if args.max_runtime is None else args.max_runtime
                ),
                "total_time_limit_seconds": (
                    "" if args.max_total_runtime is None else args.max_total_runtime
                ),
                "effective_search_limit_seconds": outcome[
                    "search_time_limit_seconds"
                ],
                "pyvrp_setup_seconds": outcome["setup_seconds"],
                "load_precision": args.load_precision,
                "load_discretization": outcome["load_discretization"],
                "effective_load_scale": outcome["effective_load_scale"],
                "load_feasibility_tolerance": outcome[
                    "load_feasibility_tolerance"
                ],
                "initial_cost": plan.cost,
                "cost": outcome["cost"],
                "solver_cost": outcome["solver_cost"],
                "objective_discretization_error": outcome[
                    "objective_discretization_error"
                ],
                "improvement_over_initial_pct": improvement_pct,
                "feasible": outcome["feasible"],
                "route_count": outcome["route_count"],
                "final_fleet_by_type": json.dumps(
                    outcome["fleet_by_type"], separators=(",", ":")
                ),
                "initial_route_count": len(plan.routes),
                "initial_fleet_by_type": json.dumps(
                    dict(sorted(fleet_by_type.items())), separators=(",", ":")
                ),
                "initial_routes": json.dumps(route_payload, separators=(",", ":")),
                "selected_augmentation": plan.augmentation_index,
                "selected_rollout": plan.rollout_index,
                "neural_seconds": neural_seconds,
                "solver_seconds": outcome["solver_seconds"],
                "pyvrp_wall_seconds": outcome["wall_seconds"],
                "wall_seconds": total_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "pyvrp_version": outcome["pyvrp_version"],
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
            }
        )
        print(
            "solved={}/{} initial={:.6f} final={:.6f} improvement={:.3f}% "
            "neural_seconds={:.3f} total_seconds={:.3f}".format(
                index + 1,
                loaded.batch_size,
                plan.cost,
                outcome["cost"],
                improvement_pct,
                neural_seconds,
                total_seconds,
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
