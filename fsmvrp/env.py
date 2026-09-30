from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .problem import ProblemBatch, dominated_vehicle_mask
from .solution import LOAD_FEASIBILITY_TOLERANCE


@dataclass
class ResetState:
    problems: ProblemBatch


@dataclass
class StepState:
    selected_count: int
    current_node: torch.Tensor
    load: torch.Tensor
    used_cost: torch.Tensor
    at_the_depot: torch.Tensor
    visited: torch.Tensor
    action_mask: torch.Tensor
    finished: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.current_node.size(0)

    @property
    def rollout_size(self) -> int:
        return self.current_node.size(1)

    @property
    def vehicle_type_num(self) -> int:
        return self.current_node.size(2)

    @property
    def graph_size(self) -> int:
        return self.action_mask.size(3)


class FSMVRPEnv:
    """Sequential route-construction environment for FSMVRP.

    There is one open route slot for every vehicle type.  A return to the depot
    closes that route; departing again opens another vehicle of the same type
    and incurs its fixed cost.  Therefore the representation allows an
    unlimited number of vehicles per type without adding a vehicle index.
    """

    def __init__(
        self,
        rollout_size: int = 1,
        prune_dominated_types: bool = False,
    ):
        if rollout_size < 1:
            raise ValueError("rollout_size must be positive.")
        self.rollout_size = rollout_size
        self.prune_dominated_types = prune_dominated_types
        self.problems: Optional[ProblemBatch] = None
        self.dominated_types: Optional[torch.Tensor] = None
        self.state: Optional[StepState] = None
        self.action_history: List[torch.Tensor] = []

    def reset(self, problems: ProblemBatch) -> Tuple[ResetState, StepState]:
        problems.validate()
        self.problems = problems
        self.dominated_types = dominated_vehicle_mask(
            problems.vehicle_capacity,
            problems.vehicle_fixed_cost,
            problems.vehicle_variable_cost,
        )
        batch_size = problems.batch_size
        type_num = problems.vehicle_type_num
        graph_size = problems.problem_size + 1
        device = problems.node_xy.device

        current_node = torch.zeros(
            batch_size, self.rollout_size, type_num, dtype=torch.long, device=device
        )
        load = problems.vehicle_capacity[:, None, :].expand(
            -1, self.rollout_size, -1
        ).clone()
        used_cost = torch.zeros_like(load)
        at_the_depot = torch.ones_like(current_node, dtype=torch.bool)
        visited = torch.zeros(
            batch_size, self.rollout_size, graph_size, dtype=torch.bool, device=device
        )
        finished = torch.zeros(
            batch_size, self.rollout_size, dtype=torch.bool, device=device
        )
        self.state = StepState(
            selected_count=0,
            current_node=current_node,
            load=load,
            used_cost=used_cost,
            at_the_depot=at_the_depot,
            visited=visited,
            action_mask=torch.empty(0, device=device),
            finished=finished,
        )
        self.state.action_mask = self._build_action_mask()
        self.action_history = []
        return ResetState(problems), self.state

    def _build_action_mask(self) -> torch.Tensor:
        if self.problems is None or self.state is None:
            raise RuntimeError("Call reset before requesting an action mask.")
        batch_size = self.problems.batch_size
        type_num = self.problems.vehicle_type_num
        graph_size = self.problems.problem_size + 1
        demand = torch.cat(
            (
                torch.zeros_like(self.problems.node_demand[:, :1]),
                self.problems.node_demand,
            ),
            dim=1,
        )
        visited = self.state.visited[:, :, None, :].expand(
            batch_size, self.rollout_size, type_num, graph_size
        )
        too_large = (
            demand[:, None, None, :]
            > self.state.load[:, :, :, None] + LOAD_FEASIBILITY_TOLERANCE
        )
        invalid = visited | too_large
        if self.prune_dominated_types:
            if self.dominated_types is None:
                raise RuntimeError("Dominance mask was not initialised.")
            invalid = invalid | self.dominated_types[:, None, :, None]
        invalid = invalid.clone()
                                                                              
                                                                             
        invalid[:, :, :, 0] = self.state.at_the_depot
                                                                            
                                                                           
                                                                          
        if self.state.finished.any():
            invalid = torch.where(
                self.state.finished[:, :, None, None],
                torch.ones_like(invalid),
                invalid,
            )
            invalid[:, :, 0, 0] = torch.where(
                self.state.finished,
                torch.zeros_like(self.state.finished),
                invalid[:, :, 0, 0],
            )
        return invalid

    def step(
        self, flat_action: torch.Tensor
    ) -> Tuple[StepState, Optional[torch.Tensor], bool]:
        if self.problems is None or self.state is None:
            raise RuntimeError("Call reset before step.")
        previous = self.state
        expected_shape = (self.problems.batch_size, self.rollout_size)
        if flat_action.shape != expected_shape:
            raise ValueError("flat_action must have shape (B, R).")

        graph_size = self.problems.problem_size + 1
        vehicle_type = torch.div(flat_action, graph_size, rounding_mode="floor")
        next_node = flat_action.remainder(graph_size)
        if (vehicle_type < 0).any() or (vehicle_type >= self.problems.vehicle_type_num).any():
            raise ValueError("Action contains an invalid vehicle-type index.")

        batch_index = torch.arange(
            self.problems.batch_size, device=flat_action.device
        )[:, None].expand_as(flat_action)
        rollout_index = torch.arange(
            self.rollout_size, device=flat_action.device
        )[None, :].expand_as(flat_action)
        if previous.action_mask[
            batch_index, rollout_index, vehicle_type, next_node
        ].any():
            raise ValueError("Action violates the current FSMVRP feasibility mask.")

        old_node = previous.current_node[
            batch_index, rollout_index, vehicle_type
        ]
        all_xy = torch.cat((self.problems.depot_xy, self.problems.node_xy), dim=1)
        demand = torch.cat(
            (
                torch.zeros_like(self.problems.node_demand[:, :1]),
                self.problems.node_demand,
            ),
            dim=1,
        )
        if self.problems.distance_matrix is None:
            old_xy = all_xy[batch_index, old_node]
            next_xy = all_xy[batch_index, next_node]
            distance = (next_xy - old_xy).pow(2).sum(dim=-1).sqrt()
        else:
            distance = self.problems.distance_matrix[
                batch_index, old_node, next_node
            ]
        variable_cost = self.problems.vehicle_variable_cost[
            batch_index, vehicle_type
        ]
        fixed_cost = self.problems.vehicle_fixed_cost[
            batch_index, vehicle_type
        ]
        departure = old_node.eq(0) & next_node.ne(0)
        increment = distance * variable_cost + departure.to(distance.dtype) * fixed_cost
        used_cost = previous.used_cost.clone()
        used_cost[batch_index, rollout_index, vehicle_type] += increment

        selected_demand = demand[batch_index, next_node]
        load = previous.load.clone()
        new_load = load[batch_index, rollout_index, vehicle_type] - selected_demand
        capacity = self.problems.vehicle_capacity[batch_index, vehicle_type]
        new_load = torch.where(next_node.eq(0), capacity, new_load)
        if (new_load < -LOAD_FEASIBILITY_TOLERANCE).any():
            raise RuntimeError("Internal error: route capacity became negative.")
        load[batch_index, rollout_index, vehicle_type] = new_load
        current_node = previous.current_node.clone()
        current_node[batch_index, rollout_index, vehicle_type] = next_node
        at_the_depot = previous.at_the_depot.clone()
        at_the_depot[batch_index, rollout_index, vehicle_type] = next_node.eq(0)

        customer_move = next_node.ne(0)
        visited = previous.visited.clone()
        visited[
            batch_index[customer_move],
            rollout_index[customer_move],
            next_node[customer_move],
        ] = True
        finished = visited[:, :, 1:].all(dim=-1)
        self.state = StepState(
            selected_count=previous.selected_count + 1,
            current_node=current_node,
            load=load,
            used_cost=used_cost,
            at_the_depot=at_the_depot,
            visited=visited,
            action_mask=torch.empty(0, device=flat_action.device),
            finished=finished,
        )
        self.state.action_mask = self._build_action_mask()
        self.action_history.append(flat_action.detach().clone())

        done = bool(self.state.finished.all().item())
        reward = -self.total_cost() if done else None
        return self.state, reward, done

    def total_cost(self) -> torch.Tensor:
        if self.problems is None or self.state is None:
            raise RuntimeError("Call reset before total_cost.")
        batch_size, rollout_size, type_num = self.state.current_node.shape
        if self.problems.distance_matrix is None:
            all_xy = torch.cat(
                (self.problems.depot_xy, self.problems.node_xy), dim=1
            )
            expanded_xy = all_xy[:, None, None, :, :].expand(
                batch_size, rollout_size, type_num, -1, -1
            )
            gather_index = self.state.current_node[:, :, :, None, None].expand(
                -1, -1, -1, 1, 2
            )
            current_xy = expanded_xy.gather(3, gather_index).squeeze(3)
            depot_xy = self.problems.depot_xy[:, None, :, :].expand(
                batch_size, rollout_size, type_num, 2
            )
            return_distance = (current_xy - depot_xy).pow(2).sum(dim=-1).sqrt()
        else:
            matrix = self.problems.distance_matrix[:, None, :, :].expand(
                batch_size, rollout_size, -1, -1
            )
            return_distance = matrix.gather(
                2,
                self.state.current_node[:, :, :, None],
            ).squeeze(-1)
        return_cost = return_distance * self.problems.vehicle_variable_cost[:, None, :]
        return (self.state.used_cost + return_cost).sum(dim=-1)
