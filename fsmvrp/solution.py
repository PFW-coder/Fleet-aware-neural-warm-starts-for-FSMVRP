"""Route-plan utilities shared by neural and hybrid FSMVRP solvers."""

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from .problem import ProblemBatch


LOAD_FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ConstructedRoute:
    """One depot-to-depot route using a zero-based vehicle type.

    Customer indices follow the problem/PyVRP convention: the depot is zero
    and customers are numbered from one through ``problem_size``.
    """

    vehicle_type: int
    visits: Tuple[int, ...]


@dataclass(frozen=True)
class ConstructedSolution:
    """Best decoded FSMVRP solution for one problem instance."""

    cost: float
    routes: Tuple[ConstructedRoute, ...]
    augmentation_index: int
    rollout_index: int


def decode_action_sequence(
    actions: Iterable[int],
    problem_size: int,
    vehicle_type_num: int,
) -> Tuple[ConstructedRoute, ...]:
    """Convert interleaved joint actions into explicit vehicle routes.

    The neural environment keeps one open route slot for every vehicle type.
    Selecting the depot closes that type's current route, and a later
    departure opens another vehicle of the same type.  Once all customers
    have been visited, any remaining actions are rollout padding and ignored.
    """
    if problem_size < 1 or vehicle_type_num < 1:
        raise ValueError("problem_size and vehicle_type_num must be positive.")

    graph_size = problem_size + 1
    open_routes: List[List[int]] = [[] for _ in range(vehicle_type_num)]
    routes: List[ConstructedRoute] = []
    visited = set()

    for flat_action in actions:
        action = int(flat_action)
        if action < 0 or action >= vehicle_type_num * graph_size:
            raise ValueError("Action is outside the joint vehicle-node space.")
        vehicle_type, node = divmod(action, graph_size)

        if len(visited) == problem_size:
            break
        if node == 0:
            if open_routes[vehicle_type]:
                routes.append(
                    ConstructedRoute(
                        vehicle_type=vehicle_type,
                        visits=tuple(open_routes[vehicle_type]),
                    )
                )
                open_routes[vehicle_type] = []
            continue

        if node in visited:
            raise ValueError("A decoded customer appears more than once.")
        visited.add(node)
        open_routes[vehicle_type].append(node)

    expected = set(range(1, problem_size + 1))
    if visited != expected:
        missing = sorted(expected - visited)
        raise ValueError("Decoded action sequence is incomplete: missing {}.".format(missing))

                                                                              
                                                                     
    for vehicle_type, visits in enumerate(open_routes):
        if visits:
            routes.append(
                ConstructedRoute(vehicle_type=vehicle_type, visits=tuple(visits))
            )
    return tuple(routes)


def validate_and_cost_solution(
    problems: ProblemBatch,
    instance_index: int,
    routes: Sequence[ConstructedRoute],
    tolerance: float = LOAD_FEASIBILITY_TOLERANCE,
) -> float:
    """Validate a route plan and independently recompute its FSMVRP cost."""
    if not 0 <= instance_index < problems.batch_size:
        raise IndexError("instance_index is outside the problem batch.")

    all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
    seen = []
    total_cost = 0.0
    for route in routes:
        if not 0 <= route.vehicle_type < problems.vehicle_type_num:
            raise ValueError("Route contains an invalid vehicle type.")
        if not route.visits:
            raise ValueError("Empty routes are not valid FSMVRP vehicles.")
        if any(node < 1 or node > problems.problem_size for node in route.visits):
            raise ValueError("Route contains an invalid customer index.")

        demand = math.fsum(
            float(problems.node_demand[instance_index, node - 1].item())
            for node in route.visits
        )
        capacity = float(
            problems.vehicle_capacity[instance_index, route.vehicle_type].item()
        )
        if demand > capacity + tolerance:
            raise ValueError("Decoded route exceeds its vehicle capacity.")

        path = (0,) + route.visits + (0,)
        distance = 0.0
        for frm, to in zip(path[:-1], path[1:]):
            if problems.distance_matrix is None:
                delta = all_xy[instance_index, frm] - all_xy[instance_index, to]
                distance += float(delta.pow(2).sum().sqrt().item())
            else:
                distance += float(
                    problems.distance_matrix[instance_index, frm, to].item()
                )

        fixed = float(
            problems.vehicle_fixed_cost[instance_index, route.vehicle_type].item()
        )
        variable = float(
            problems.vehicle_variable_cost[instance_index, route.vehicle_type].item()
        )
        total_cost += fixed + variable * distance
        seen.extend(route.visits)

    if sorted(seen) != list(range(1, problems.problem_size + 1)):
        raise ValueError("Routes must visit every customer exactly once.")
    return total_cost
