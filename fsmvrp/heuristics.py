"""Deterministic non-learning construction heuristics for FSMVRP."""

import math
from typing import List, Sequence, Tuple

import torch

from .problem import ProblemBatch
from .solution import (
    LOAD_FEASIBILITY_TOLERANCE,
    ConstructedRoute,
    ConstructedSolution,
    validate_and_cost_solution,
)


def _distance_matrix(problems: ProblemBatch, instance_index: int) -> torch.Tensor:
    if problems.distance_matrix is not None:
        return problems.distance_matrix[instance_index].detach().cpu()
    xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
    return torch.cdist(xy[instance_index].detach().cpu(), xy[instance_index].detach().cpu())


def _candidate_sweep_orders(
    problems: ProblemBatch,
    instance_index: int,
    order_count: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Return deterministic angular orders with multiple cuts and directions."""
    if order_count <= 0:
        raise ValueError("order_count must be positive.")

    depot = problems.depot_xy[instance_index, 0].detach().cpu()
    customer_xy = problems.node_xy[instance_index].detach().cpu()
    relative = customer_xy - depot
    angles = torch.atan2(relative[:, 1], relative[:, 0])
    radii = relative.pow(2).sum(dim=1)
    base = tuple(
        sorted(
            range(1, problems.problem_size + 1),
            key=lambda customer: (
                float(angles[customer - 1]),
                float(radii[customer - 1]),
                customer,
            ),
        )
    )

    candidates: List[Tuple[int, ...]] = []
    seen = set()
                                                                           
                                                                            
    cut_count = min(len(base), max(1, math.ceil(order_count / 2)))
    cuts = [int(math.floor(index * len(base) / cut_count)) for index in range(cut_count)]
    for cut in cuts:
        rotated = base[cut:] + base[:cut]
        for order in (rotated, tuple(reversed(rotated))):
            if order not in seen:
                candidates.append(order)
                seen.add(order)
            if len(candidates) == order_count:
                return tuple(candidates)
    return tuple(candidates)


def _best_partition_for_order(
    problems: ProblemBatch,
    instance_index: int,
    order: Sequence[int],
    matrix: torch.Tensor,
) -> Tuple[ConstructedRoute, ...]:
    """Partition one sweep order by dynamic programming.

    Every contiguous segment is assigned the cheapest capacity-feasible
    vehicle type for that segment's depot-to-depot route.  The dynamic program
    then selects the minimum-cost segmentation of the complete angular order.
    """
    customer_count = len(order)
    demands = problems.node_demand[instance_index].detach().cpu().tolist()
    capacities = problems.vehicle_capacity[instance_index].detach().cpu().tolist()
    fixed_costs = problems.vehicle_fixed_cost[instance_index].detach().cpu().tolist()
    variable_costs = problems.vehicle_variable_cost[instance_index].detach().cpu().tolist()
    maximum_capacity = max(capacities)

    best_cost = [math.inf] * (customer_count + 1)
    predecessor: List[Tuple[int, int] | None] = [None] * (customer_count + 1)
    best_cost[0] = 0.0

    for start in range(customer_count):
        if not math.isfinite(best_cost[start]):
            continue
        first = order[start]
        route_demand = 0.0
        route_distance = 0.0
        previous = 0
        for stop in range(start, customer_count):
            customer = order[stop]
            route_demand += float(demands[customer - 1])
            if route_demand > maximum_capacity + LOAD_FEASIBILITY_TOLERANCE:
                break

            if stop == start:
                route_distance = float(matrix[0, first]) + float(matrix[first, 0])
            else:
                route_distance += (
                    float(matrix[previous, customer])
                    + float(matrix[customer, 0])
                    - float(matrix[previous, 0])
                )
            previous = customer

            type_costs = (
                (
                    float(fixed_costs[type_index])
                    + float(variable_costs[type_index]) * route_distance,
                    type_index,
                )
                for type_index, capacity in enumerate(capacities)
                if route_demand <= float(capacity) + LOAD_FEASIBILITY_TOLERANCE
            )
            segment_cost, vehicle_type = min(type_costs)
            candidate = best_cost[start] + segment_cost
            if candidate < best_cost[stop + 1] - 1e-12:
                best_cost[stop + 1] = candidate
                predecessor[stop + 1] = (start, vehicle_type)

    if predecessor[-1] is None:
        raise RuntimeError("The sweep dynamic program could not construct a feasible solution.")

    routes: List[ConstructedRoute] = []
    cursor = customer_count
    while cursor:
        parent = predecessor[cursor]
        if parent is None:
            raise RuntimeError("The sweep dynamic-programming predecessor chain is incomplete.")
        start, vehicle_type = parent
        routes.append(
            ConstructedRoute(
                vehicle_type=vehicle_type,
                visits=tuple(order[start:cursor]),
            )
        )
        cursor = start
    routes.reverse()
    return tuple(routes)


def construct_sweep_solution(
    problems: ProblemBatch,
    instance_index: int,
    order_count: int = 8,
) -> ConstructedSolution:
    """Construct a type-aware sweep solution without learned parameters.

    The baseline orders customers by angle around the depot, considers several
    circular cuts in both directions, and optimally partitions each order into
    capacity-feasible routes while choosing the least-cost vehicle type for
    every route.  It is deterministic and uses only the instance data.
    """
    if not 0 <= instance_index < problems.batch_size:
        raise IndexError("instance_index is outside the problem batch.")
    problems.validate()
    matrix = _distance_matrix(problems, instance_index)
    candidates = _candidate_sweep_orders(problems, instance_index, order_count)
    if not candidates:
        raise RuntimeError("No sweep order was generated.")

    best_cost = math.inf
    best_routes: Tuple[ConstructedRoute, ...] = ()
    best_order_index = -1
    for order_index, order in enumerate(candidates):
        routes = _best_partition_for_order(problems, instance_index, order, matrix)
        cost = validate_and_cost_solution(problems, instance_index, routes)
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_routes = routes
            best_order_index = order_index

    return ConstructedSolution(
        cost=best_cost,
        routes=best_routes,
        augmentation_index=-1,
        rollout_index=best_order_index,
    )
