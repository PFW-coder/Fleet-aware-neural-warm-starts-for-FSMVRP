"""Run the PyVRP metaheuristic under an explicit per-instance time limit."""

import argparse
import csv
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch

from fsmvrp.problem import ProblemBatch, load_problems
from fsmvrp.solution import (
    LOAD_FEASIBILITY_TOLERANCE,
    ConstructedRoute,
    validate_and_cost_solution,
)


def distance_matrix(problems: ProblemBatch, index: int) -> torch.Tensor:
    if problems.distance_matrix is not None:
        return problems.distance_matrix[index].cpu()
    xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)[index].cpu()
    return torch.cdist(xy, xy, p=2)


def discretized_arc_cost(
    matrix: torch.Tensor,
    frm_index: int,
    to_index: int,
    variable_cost: float,
    objective_precision: int,
) -> int:
    """Return PyVRP's integer arc cost with an exact zero diagonal.

    Distance matrices computed in floating point can contain tiny positive
    diagonal residuals. PyVRP rejects non-zero self loops, while self loops
    are not part of any FSMVRP route, so set them to their exact mathematical
    value before discretisation.
    """
    if frm_index == to_index:
        return 0
    return int(
        round(
            variable_cost
            * float(matrix[frm_index, to_index])
            * objective_precision
        )
    )


def discretized_arc_cost_matrix(
    matrix: torch.Tensor,
    variable_cost: float,
    objective_precision: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Discretise one vehicle type's complete arc-cost matrix.

    Converting the distance tensor to nested Python lists once avoids millions
    of scalar tensor indexing operations while preserving the exact arithmetic
    and rounding order used by :func:`discretized_arc_cost`.
    """
    rows = matrix.tolist()
    return tuple(
        tuple(
            0
            if frm_index == to_index
            else int(
                round(
                    variable_cost
                    * float(distance)
                    * objective_precision
                )
            )
            for to_index, distance in enumerate(row)
        )
        for frm_index, row in enumerate(rows)
    )


def exact_binary_load_scale(
    values: Iterable[float], minimum_precision: int
) -> int:
    """Return a power-of-two scale that represents every stored load exactly.

    PyTorch problem tensors store IEEE binary floating-point numbers.  Their
    exact denominators are powers of two, so the largest denominator is a
    common scale.  A power-of-two lower bound also honours the requested
    minimum precision without reintroducing decimal rounding error.
    """
    if minimum_precision <= 0:
        raise ValueError("minimum_precision must be positive.")
    scale = 1 << (int(minimum_precision) - 1).bit_length()
    for value in values:
        denominator = float(value).as_integer_ratio()[1]
        scale = max(scale, denominator)
    return scale


def exact_scaled_load(value: float, load_scale: int) -> int:
    """Map an IEEE floating-point load exactly to a common integer scale."""
    numerator, denominator = float(value).as_integer_ratio()
    multiplier, remainder = divmod(load_scale, denominator)
    if remainder:
        raise ValueError("load_scale does not exactly represent this load value.")
    return numerator * multiplier


def discretized_demand(value: float, load_scale: int) -> int:
    return exact_scaled_load(value, load_scale)


def discretized_capacity(
    value: float,
    load_scale: int,
    tolerance: float = LOAD_FEASIBILITY_TOLERANCE,
) -> int:
    """Map capacity exactly and add only the declared feasibility tolerance."""
    return exact_scaled_load(value, load_scale) + math.floor(tolerance * load_scale)


def build_model(
    problems: ProblemBatch,
    index: int,
    objective_precision: int,
    load_precision: int,
):
    try:
        from pyvrp import Model
    except ImportError as error:
        raise RuntimeError(
            "PyVRP is not installed. Use requirements-baseline.txt in Python 3.11+."
        ) from error

    load_values = tuple(problems.node_demand[index].tolist()) + tuple(
        problems.vehicle_capacity[index].tolist()
    )
    load_scale = exact_binary_load_scale(load_values, load_precision)
    matrix = distance_matrix(problems, index)
    all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)[index].cpu()
    model = Model()
    locations = [
        model.add_location(
            x=int(round(float(point[0]) * 1000)),
            y=int(round(float(point[1]) * 1000)),
        )
        for point in all_xy
    ]
    model.add_depot(location=locations[0], name="depot")
    for customer_index in range(problems.problem_size):
        delivery = discretized_demand(
            problems.node_demand[index, customer_index], load_scale
        )
        model.add_client(
            location=locations[customer_index + 1],
            delivery=delivery,
            name="customer_{}".format(customer_index + 1),
        )

    for type_index in range(problems.vehicle_type_num):
        profile = model.add_profile(name="vehicle_type_{}".format(type_index))
        variable_cost = float(problems.vehicle_variable_cost[index, type_index])
        arc_costs = discretized_arc_cost_matrix(
            matrix,
            variable_cost,
            objective_precision,
        )
        for frm_index, frm in enumerate(locations):
            for to_index, to in enumerate(locations):
                profile.add_edge(
                    frm,
                    to,
                    distance=arc_costs[frm_index][to_index],
                )
        model.add_vehicle_type(
            num_available=problems.problem_size,
            capacity=discretized_capacity(
                problems.vehicle_capacity[index, type_index], load_scale
            ),
            fixed_cost=int(
                round(
                    float(problems.vehicle_fixed_cost[index, type_index])
                    * objective_precision
                )
            ),
            unit_distance_cost=1,
            profile=profile,
            name="vehicle_type_{}".format(type_index),
        )
    return model, load_scale


def make_initial_solution(model, routes: Sequence[ConstructedRoute]):
    """Convert a neural FSMVRP route plan into a PyVRP solution.

    Neural routes use zero for the depot and one-based customer identifiers.
    PyVRP's ``Route`` constructor instead numbers clients from zero, so every
    visit must be shifted down by one at this interface boundary.
    """
    try:
        from pyvrp import Route, Solution
    except ImportError as error:
        raise RuntimeError(
            "PyVRP is not installed. Use requirements-baseline.txt in Python 3.11+."
        ) from error

    data = model.data()
    pyvrp_routes = [
        Route(
            data,
            [client - 1 for client in route.visits],
            route.vehicle_type,
        )
        for route in routes
    ]
    solution = Solution(data, pyvrp_routes)
    if not solution.is_feasible():
        raise ValueError(
            "The neural route plan is infeasible in the PyVRP model: "
            "excess_load={}, excess_distance={}, time_warp={}, complete={}.".format(
                solution.excess_load(),
                solution.excess_distance(),
                solution.time_warp(),
                solution.is_complete(),
            )
        )
    return solution


def extract_solution_routes(solution) -> Tuple[ConstructedRoute, ...]:
    """Convert a PyVRP solution back to the neural problem's route indexing."""
    return tuple(
        ConstructedRoute(
            vehicle_type=int(route.vehicle_type()),
            visits=tuple(
                int(activity.idx) + 1
                for activity in route
                if activity.is_client()
            ),
        )
        for route in solution.routes()
    )


def solve_instance(
    problems: ProblemBatch,
    index: int,
    max_runtime: Optional[float],
    seed: int,
    objective_precision: int,
    load_precision: int,
    initial_routes: Optional[Sequence[ConstructedRoute]] = None,
    max_wall_runtime: Optional[float] = None,
):
    if (max_runtime is None) == (max_wall_runtime is None):
        raise ValueError(
            "Specify exactly one of max_runtime and max_wall_runtime."
        )
    if max_runtime is not None and max_runtime <= 0:
        raise ValueError("max_runtime must be positive.")
    if max_wall_runtime is not None and max_wall_runtime <= 0:
        raise ValueError("max_wall_runtime must be positive.")

    try:
        import pyvrp
        from pyvrp.stop import MaxRuntime
    except ImportError as error:
        raise RuntimeError(
            "PyVRP is not installed. Use requirements-baseline.txt in Python 3.11+."
        ) from error

    wall_start = time.perf_counter()
    model, effective_load_scale = build_model(
        problems,
        index,
        objective_precision=objective_precision,
        load_precision=load_precision,
    )
    initial_solution = (
        None if initial_routes is None else make_initial_solution(model, initial_routes)
    )
    setup_seconds = time.perf_counter() - wall_start
    search_time_limit = max_runtime
    if max_wall_runtime is not None:
        search_time_limit = max_wall_runtime - setup_seconds
        if search_time_limit <= 0:
            raise RuntimeError(
                "The PyVRP model setup used {:.6f}s, exceeding the {:.6f}s "
                "wall-time budget.".format(setup_seconds, max_wall_runtime)
            )

    result = model.solve(
        stop=MaxRuntime(search_time_limit),
        seed=seed,
        collect_stats=False,
        display=False,
        initial_solution=initial_solution,
    )
    wall_seconds = time.perf_counter() - wall_start
    feasible = bool(result.is_feasible())
    solver_cost = (
        float(result.cost()) / objective_precision if feasible else float("inf")
    )
    if feasible:
        final_routes = extract_solution_routes(result.best)
        try:
            cost = validate_and_cost_solution(problems, index, final_routes)
        except ValueError as error:
            raise RuntimeError(
                "PyVRP returned a route plan that is infeasible under the original "
                "continuous FSMVRP data outside the declared load tolerance "
                "despite exact binary load mapping. Inspect the PyVRP/FSMVRP "
                "route interface."
            ) from error
    else:
        final_routes = ()
        cost = float("inf")
    fleet_by_type = Counter(route.vehicle_type for route in final_routes)
    return {
        "cost": cost,
        "solver_cost": solver_cost,
        "objective_discretization_error": cost - solver_cost,
        "feasible": int(feasible),
        "route_count": len(final_routes),
        "fleet_by_type": dict(sorted(fleet_by_type.items())),
        "solver_seconds": float(getattr(result, "runtime", search_time_limit)),
        "setup_seconds": setup_seconds,
        "search_time_limit_seconds": search_time_limit,
        "wall_time_limit_seconds": (
            "" if max_wall_runtime is None else max_wall_runtime
        ),
        "wall_seconds": wall_seconds,
        "load_discretization": "exact_binary_with_tolerance",
        "effective_load_scale": effective_load_scale,
        "load_feasibility_tolerance": LOAD_FEASIBILITY_TOLERANCE,
        "pyvrp_version": _pyvrp_version(pyvrp),
    }


def _pyvrp_version(pyvrp_module) -> str:
    module_version = getattr(pyvrp_module, "__version__", None)
    if module_version is not None:
        return str(module_version)
    try:
        return version("pyvrp")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PyVRP baseline.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--method-name")
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--max-runtime",
        type=float,
        help="PyVRP search time, excluding model construction.",
    )
    time_group.add_argument(
        "--max-total-runtime",
        type=float,
        help="Per-instance wall-time budget including PyVRP model construction.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective-precision", type=int, default=10000)
    parser.add_argument("--load-precision", type=int, default=1000000000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_limit = (
        args.max_runtime
        if args.max_runtime is not None
        else args.max_total_runtime
    )
    if requested_limit <= 0:
        raise ValueError("The selected runtime limit must be positive.")
    if args.objective_precision <= 0 or args.load_precision <= 0:
        raise ValueError("Precision factors must be positive.")
    problems = load_problems(str(args.data), torch.device("cpu"))
    dataset_name = args.dataset_name or args.data.stem
    rows = []
    for index in range(problems.batch_size):
        outcome = solve_instance(
            problems,
            index,
            args.max_runtime,
            args.seed,
            args.objective_precision,
            args.load_precision,
            max_wall_runtime=args.max_total_runtime,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "instance_id": index,
                "method": args.method_name or "pyvrp_{}s".format(args.max_runtime),
                "solver_seed": args.seed,
                "problem_size": problems.problem_size,
                "vehicle_types": problems.vehicle_type_num,
                "time_limit_seconds": requested_limit,
                "time_limit_mode": (
                    "search" if args.max_runtime is not None else "total_wall"
                ),
                "effective_search_limit_seconds": outcome[
                    "search_time_limit_seconds"
                ],
                "setup_seconds": outcome["setup_seconds"],
                "objective_precision": args.objective_precision,
                "load_precision": args.load_precision,
                "load_discretization": outcome["load_discretization"],
                "effective_load_scale": outcome["effective_load_scale"],
                "load_feasibility_tolerance": outcome[
                    "load_feasibility_tolerance"
                ],
                "cost": outcome["cost"],
                "solver_cost": outcome["solver_cost"],
                "objective_discretization_error": outcome[
                    "objective_discretization_error"
                ],
                "feasible": outcome["feasible"],
                "route_count": outcome["route_count"],
                "fleet_by_type": json.dumps(
                    outcome["fleet_by_type"], separators=(",", ":")
                ),
                "solver_seconds": outcome["solver_seconds"],
                "wall_seconds": outcome["wall_seconds"],
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "pyvrp_version": outcome["pyvrp_version"],
                "python_version": platform.python_version(),
            }
        )
        print(
            "solved={}/{} cost={:.6f} feasible={}".format(
                index + 1,
                problems.batch_size,
                outcome["cost"],
                outcome["feasible"],
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
