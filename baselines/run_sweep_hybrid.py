"""Type-aware sweep construction followed by bounded PyVRP improvement."""

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
from fsmvrp.heuristics import construct_sweep_solution
from fsmvrp.problem import load_problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-learning sweep-DP warm-start baseline."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--method-name", default="sweep_dp_pyvrp")
    parser.add_argument("--sweep-orders", type=int, default=8)
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--max-runtime",
        type=float,
        help="PyVRP search time after sweep construction.",
    )
    time_group.add_argument(
        "--max-total-runtime",
        type=float,
        help="Per-instance wall-time budget including construction and PyVRP.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective-precision", type=int, default=1000000)
    parser.add_argument("--load-precision", type=int, default=1000000000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_limit = (
        args.max_runtime if args.max_runtime is not None else args.max_total_runtime
    )
    if requested_limit <= 0:
        raise ValueError("The selected runtime limit must be positive.")
    if args.sweep_orders <= 0:
        raise ValueError("sweep-orders must be positive.")
    if args.objective_precision <= 0 or args.load_precision <= 0:
        raise ValueError("Precision factors must be positive.")

    problems = load_problems(str(args.data), torch.device("cpu"))
    dataset_name = args.dataset_name or args.data.stem
    rows = []
    for index in range(problems.batch_size):
        total_start = time.perf_counter()
        construction_start = time.perf_counter()
        plan = construct_sweep_solution(
            problems,
            instance_index=index,
            order_count=args.sweep_orders,
        )
        construction_seconds = time.perf_counter() - construction_start

        remaining_wall_runtime = None
        if args.max_total_runtime is not None:
            elapsed_before_pyvrp = time.perf_counter() - total_start
            remaining_wall_runtime = args.max_total_runtime - elapsed_before_pyvrp
            if remaining_wall_runtime <= 0:
                raise RuntimeError(
                    "Sweep construction used {:.6f}s, exceeding the {:.6f}s "
                    "total wall-time budget.".format(
                        elapsed_before_pyvrp, args.max_total_runtime
                    )
                )

        outcome = solve_instance(
            problems,
            index,
            max_runtime=args.max_runtime,
            seed=args.seed,
            objective_precision=args.objective_precision,
            load_precision=args.load_precision,
            initial_routes=plan.routes,
            max_wall_runtime=remaining_wall_runtime,
        )
        total_seconds = time.perf_counter() - total_start
        improvement_pct = 100.0 * (plan.cost - outcome["cost"]) / plan.cost
        initial_fleet = Counter(route.vehicle_type for route in plan.routes)
        initial_routes = [
            {"vehicle_type": route.vehicle_type, "visits": list(route.visits)}
            for route in plan.routes
        ]
        rows.append(
            {
                "dataset": dataset_name,
                "instance_id": index,
                "method": args.method_name,
                "solver_seed": args.seed,
                "problem_size": problems.problem_size,
                "vehicle_types": problems.vehicle_type_num,
                "constructor": "angular_sweep_dp",
                "sweep_orders": args.sweep_orders,
                "selected_sweep_order": plan.rollout_index,
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
                "objective_precision": args.objective_precision,
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
                    dict(sorted(initial_fleet.items())), separators=(",", ":")
                ),
                "initial_routes": json.dumps(initial_routes, separators=(",", ":")),
                "construction_seconds": construction_seconds,
                "solver_seconds": outcome["solver_seconds"],
                "pyvrp_wall_seconds": outcome["wall_seconds"],
                "wall_seconds": total_seconds,
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "pyvrp_version": outcome["pyvrp_version"],
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
            }
        )
        print(
            "solved={}/{} initial={:.6f} final={:.6f} improvement={:.3f}% "
            "sweep_seconds={:.3f} total_seconds={:.3f}".format(
                index + 1,
                problems.batch_size,
                plan.cost,
                outcome["cost"],
                improvement_pct,
                construction_seconds,
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
