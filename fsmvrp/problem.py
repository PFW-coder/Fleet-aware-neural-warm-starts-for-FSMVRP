import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ProblemBatch:
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor
    vehicle_capacity: torch.Tensor
    vehicle_fixed_cost: torch.Tensor
    vehicle_variable_cost: torch.Tensor
                                                                              
                                                                       
    distance_matrix: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self.node_xy.size(0)

    @property
    def problem_size(self) -> int:
        return self.node_xy.size(1)

    @property
    def vehicle_type_num(self) -> int:
        return self.vehicle_capacity.size(1)

    def to(self, device: torch.device) -> "ProblemBatch":
        return ProblemBatch(
            **{
                key: None if value is None else value.to(device)
                for key, value in asdict(self).items()
            }
        )

    def slice(self, start: int, stop: int) -> "ProblemBatch":
        """Return a contiguous subset without changing the instance fields."""
        return ProblemBatch(
            **{
                key: None if value is None else value[start:stop]
                for key, value in asdict(self).items()
            }
        )

    def validate(self) -> None:
        batch_size, problem_size, coordinate_dim = self.node_xy.shape
        if coordinate_dim != 2 or self.depot_xy.shape != (batch_size, 1, 2):
            raise ValueError("Coordinates must have shapes (B, N, 2) and (B, 1, 2).")
        if self.node_demand.shape != (batch_size, problem_size):
            raise ValueError("node_demand must have shape (B, N).")
        vehicle_shape = self.vehicle_capacity.shape
        if len(vehicle_shape) != 2 or vehicle_shape[0] != batch_size:
            raise ValueError("vehicle_capacity must have shape (B, K).")
        if self.vehicle_fixed_cost.shape != vehicle_shape or self.vehicle_variable_cost.shape != vehicle_shape:
            raise ValueError("All vehicle attributes must have shape (B, K).")
        if self.distance_matrix is not None:
            expected = (batch_size, problem_size + 1, problem_size + 1)
            if self.distance_matrix.shape != expected:
                raise ValueError(
                    "distance_matrix must have shape (B, N + 1, N + 1)."
                )
            if (self.distance_matrix < 0).any():
                raise ValueError("Distances must be non-negative.")
        tensors = [value for value in asdict(self).values() if value is not None]
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Problem data must contain only finite values.")
        if (self.node_demand <= 0).any():
            raise ValueError("Every customer demand must be positive.")
        if (self.vehicle_capacity <= 0).any():
            raise ValueError("Every vehicle capacity must be positive.")
        if (self.vehicle_fixed_cost < 0).any() or (self.vehicle_variable_cost < 0).any():
            raise ValueError("Vehicle costs must be non-negative.")
        max_capacity = self.vehicle_capacity.max(dim=1).values[:, None]
        if (self.node_demand > max_capacity + 1e-8).any():
            raise ValueError(
                "Infeasible instance: a customer demand exceeds every vehicle capacity."
            )


def dominated_vehicle_mask(
    capacity: torch.Tensor,
    fixed_cost: torch.Tensor,
    variable_cost: torch.Tensor,
) -> torch.Tensor:
    """Return True for a type weakly dominated by another type in the same instance."""
    cap_i = capacity[:, :, None]
    cap_j = capacity[:, None, :]
    fixed_i = fixed_cost[:, :, None]
    fixed_j = fixed_cost[:, None, :]
    variable_i = variable_cost[:, :, None]
    variable_j = variable_cost[:, None, :]
    weak = (cap_j >= cap_i) & (fixed_j <= fixed_i) & (variable_j <= variable_i)
    strict = (cap_j > cap_i) | (fixed_j < fixed_i) | (variable_j < variable_i)
    return (weak & strict).any(dim=-1)


def _sample_vehicle_attributes(
    batch_size: int,
    vehicle_type_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capacity = torch.rand(batch_size, vehicle_type_num, device=device, dtype=dtype) * 2.5 + 0.5
    mean_factor = torch.rand(batch_size, 1, device=device, dtype=dtype) * 19.0 + 1.0
    fixed_factor = torch.clamp(
        torch.rand(batch_size, vehicle_type_num, device=device, dtype=dtype) * 2.0 - 1.0 + mean_factor,
        min=1.0,
    )
    fixed_cost = fixed_factor * capacity
    variable_cost = torch.rand(batch_size, vehicle_type_num, device=device, dtype=dtype) * 2.0 + 1.0
    return capacity, fixed_cost, variable_cost


def generate_problems(
    batch_size: int,
    problem_size: int,
    vehicle_type_num: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    ensure_nondominated: bool = False,
    max_resample_attempts: int = 200,
) -> ProblemBatch:
    """Generate the paper's synthetic FSMVRP distribution.

    ``ensure_nondominated`` is intended for a controlled robustness split.  The
    default remains the original unfiltered distribution so results are not
    silently changed.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    depot_xy = torch.rand(batch_size, 1, 2, device=device, dtype=dtype)
    node_xy = torch.rand(batch_size, problem_size, 2, device=device, dtype=dtype)
    node_demand = torch.randint(
        1, 51, (batch_size, problem_size), device=device
    ).to(dtype) / 100.0
    capacity, fixed_cost, variable_cost = _sample_vehicle_attributes(
        batch_size, vehicle_type_num, device, dtype
    )

    if ensure_nondominated:
        rejected = dominated_vehicle_mask(capacity, fixed_cost, variable_cost).any(dim=1)
        attempts = 0
        while rejected.any() and attempts < max_resample_attempts:
            count = int(rejected.sum().item())
            new_capacity, new_fixed, new_variable = _sample_vehicle_attributes(
                count, vehicle_type_num, device, dtype
            )
            capacity[rejected] = new_capacity
            fixed_cost[rejected] = new_fixed
            variable_cost[rejected] = new_variable
            rejected = dominated_vehicle_mask(capacity, fixed_cost, variable_cost).any(dim=1)
            attempts += 1
        if rejected.any():
            raise RuntimeError("Could not sample a nondominated vehicle set.")

    problems = ProblemBatch(
        depot_xy=depot_xy,
        node_xy=node_xy,
        node_demand=node_demand,
        vehicle_capacity=capacity,
        vehicle_fixed_cost=fixed_cost,
        vehicle_variable_cost=variable_cost,
    )
    problems.validate()
    return problems


def _sample_fsmfd_locations(
    batch_size: int,
    problem_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample uniform, clustered, or mixed locations and their scale."""
    depot_xy = torch.rand(batch_size, 1, 2, device=device, dtype=dtype)
    uniform_xy = torch.rand(
        batch_size, problem_size, 2, device=device, dtype=dtype
    )

    max_clusters = 8
    centers = torch.rand(batch_size, max_clusters, 2, device=device, dtype=dtype)
    cluster_count = torch.randint(3, max_clusters + 1, (batch_size, 1), device=device)
    assignment = (
        torch.rand(batch_size, problem_size, device=device) * cluster_count
    ).long()
    clustered_xy = centers.gather(
        1, assignment[:, :, None].expand(-1, -1, 2)
    )
    cluster_sigma = (
        torch.rand(batch_size, 1, 1, device=device, dtype=dtype) * 0.09 + 0.03
    )
    clustered_xy = (
        clustered_xy
        + torch.randn(
            batch_size, problem_size, 2, device=device, dtype=dtype
        )
        * cluster_sigma
    ).clamp(0.0, 1.0)
    mixed_xy = torch.where(
        torch.rand(batch_size, problem_size, 1, device=device) < 0.5,
        uniform_xy,
        clustered_xy,
    )
    spatial_family = torch.randint(0, 3, (batch_size, 1, 1), device=device)
    node_xy = torch.where(
        spatial_family == 0,
        uniform_xy,
        torch.where(spatial_family == 1, clustered_xy, mixed_xy),
    )

    all_xy = torch.cat((depot_xy, node_xy), dim=1)
    coordinate_range = all_xy.max(dim=1).values - all_xy.min(dim=1).values
    coordinate_scale = coordinate_range.max(dim=1).values.clamp_min(1e-8)
    return depot_xy, node_xy, coordinate_scale


def _ordered_fleet_position(
    batch_size: int,
    vehicle_type_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if vehicle_type_num == 1:
        position = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    else:
        position = torch.linspace(
            0.0, 1.0, vehicle_type_num, device=device, dtype=dtype
        )[None, :]
    return position.expand(batch_size, -1)


def generate_fsmfd_problems(
    batch_size: int,
    problem_size: int,
    vehicle_type_num: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> ProblemBatch:
    """Generate scale-normalised, benchmark-shaped FSMFD instances.

    The public FSMFD instances differ from the original synthetic generator in
    two important ways: their customer layouts include clustered and mixed
    spatial patterns, and vehicle capacity, fixed cost, and distance cost form
    ordered trade-offs.  This generator reproduces those broad structural
    properties without using benchmark coordinates or solutions.

    Costs are constructed through the dimensionless ratio
    ``fixed_cost / (variable_cost * coordinate_scale)``.  Sampling that ratio
    directly keeps the distribution unchanged under a consistent change of
    distance units and matches the normalisation used by the model.
    """
    if batch_size < 1 or problem_size < 1 or vehicle_type_num < 1:
        raise ValueError("batch_size, problem_size, and vehicle_type_num must be positive.")
    device = torch.device("cpu") if device is None else torch.device(device)

                                                                             
                                                                
    depot_xy, node_xy, coordinate_scale = _sample_fsmfd_locations(
        batch_size, problem_size, device, dtype
    )
    position = _ordered_fleet_position(
        batch_size, vehicle_type_num, device, dtype
    )

    max_capacity = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.4 + 0.8
    min_capacity_ratio = (
        torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.32 + 0.18
    )
    capacity = max_capacity * min_capacity_ratio.pow(1.0 - position)

    variable_min = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.3 + 0.4
    variable_growth = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.5 + 2.0
    variable_cost = variable_min * variable_growth.pow(position)

    fixed_ratio_min = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.55 + 0.25
    fixed_ratio_max = torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 + 1.0
    fixed_travel_ratio = fixed_ratio_min * (
        fixed_ratio_max / fixed_ratio_min
    ).pow(position)
    fixed_cost = (
        fixed_travel_ratio * variable_cost * coordinate_scale[:, None]
    )

                                                                              
                                                                              
                                                                             
                                                                             
    demand_family = torch.randint(0, 3, (batch_size,), device=device)
    demand_draw = torch.rand(
        batch_size, problem_size, device=device, dtype=dtype
    )
    demand_weight = torch.where(
        demand_family[:, None] == 0,
        0.5 + 0.5 * demand_draw,
        torch.where(
            demand_family[:, None] == 1,
            0.1 + 0.9 * demand_draw,
            0.01 + 0.99 * demand_draw.square(),
        ),
    )
    total_low = min(6.0, 0.10 * problem_size)
    total_high = min(24.0, 0.18 * problem_size)
    target_total = (
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * (total_high - total_low)
        + total_low
    )
    demand_ratio = demand_weight / demand_weight.sum(dim=1, keepdim=True)
    demand_ratio = (demand_ratio * target_total).clamp(max=0.5)
    node_demand = demand_ratio * max_capacity

    problems = ProblemBatch(
        depot_xy=depot_xy,
        node_xy=node_xy,
        node_demand=node_demand,
        vehicle_capacity=capacity,
        vehicle_fixed_cost=fixed_cost,
        vehicle_variable_cost=variable_cost,
    )
    problems.validate()
    return problems


def generate_fsmfd_v2_problems(
    batch_size: int,
    problem_size: int,
    vehicle_type_num: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> ProblemBatch:
    """Generate FSMFD instances with broader fleet-level operating regimes.

    The first benchmark-shaped generator covers the pooled public attribute
    range, but forces every individual fleet to cross from a low to a high
    fixed-to-travel-cost ratio.  Public FSMFD fleets may instead be entirely
    low-cost or entirely high-cost.  This expanded distribution samples an
    instance-level cost regime separately from its within-fleet spread.  It
    also covers wider capacity and variable-cost spans and adds a heavy-tailed
    demand family.  Coordinates and solutions from public instances are never
    used.
    """
    if batch_size < 1 or problem_size < 1 or vehicle_type_num < 1:
        raise ValueError("batch_size, problem_size, and vehicle_type_num must be positive.")
    device = torch.device("cpu") if device is None else torch.device(device)

    depot_xy, node_xy, coordinate_scale = _sample_fsmfd_locations(
        batch_size, problem_size, device, dtype
    )
    position = _ordered_fleet_position(
        batch_size, vehicle_type_num, device, dtype
    )

    max_capacity = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.4 + 0.8
    capacity_span = torch.exp(
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * math.log(6.2 / 1.8)
        + math.log(1.8)
    )
    capacity = max_capacity / capacity_span.pow(1.0 - position)

    variable_min = torch.rand(batch_size, 1, device=device, dtype=dtype) * 0.4 + 0.35
    variable_span = torch.exp(
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * math.log(6.2 / 1.5)
        + math.log(1.5)
    )
    variable_cost = variable_min * variable_span.pow(position)

                                                                         
                                                                               
    ratio_center = torch.exp(
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * math.log(1.9 / 0.3)
        + math.log(0.3)
    )
    ratio_span = torch.exp(
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * math.log(2.5 / 1.2)
        + math.log(1.2)
    )
    ratio_root = ratio_span.sqrt()
    ratio_min = (ratio_center / ratio_root).clamp(min=0.2, max=3.2)
    ratio_max = (ratio_center * ratio_root).clamp(min=0.2, max=3.2)
    fixed_travel_ratio = ratio_min * (
        ratio_max / ratio_min.clamp_min(1e-8)
    ).pow(position)
    fixed_cost = fixed_travel_ratio * variable_cost * coordinate_scale[:, None]

    demand_family = torch.randint(0, 4, (batch_size,), device=device)
    demand_draw = torch.rand(
        batch_size, problem_size, device=device, dtype=dtype
    )
    demand_weight = torch.where(
        demand_family[:, None] == 0,
        0.5 + 0.5 * demand_draw,
        torch.where(
            demand_family[:, None] == 1,
            0.1 + 0.9 * demand_draw,
            torch.where(
                demand_family[:, None] == 2,
                0.01 + 0.99 * demand_draw.square(),
                0.001 + 0.999 * demand_draw.pow(6),
            ),
        ),
    )
    total_low = min(6.0, 0.10 * problem_size)
    total_high = min(26.0, 0.22 * problem_size)
    target_total = (
        torch.rand(batch_size, 1, device=device, dtype=dtype)
        * (total_high - total_low)
        + total_low
    )
    demand_ratio = demand_weight / demand_weight.sum(dim=1, keepdim=True)
    demand_ratio = (demand_ratio * target_total).clamp(max=0.5)
    node_demand = demand_ratio * max_capacity

    problems = ProblemBatch(
        depot_xy=depot_xy,
        node_xy=node_xy,
        node_demand=node_demand,
        vehicle_capacity=capacity,
        vehicle_fixed_cost=fixed_cost,
        vehicle_variable_cost=variable_cost,
    )
    problems.validate()
    return problems


def save_problems(problems: ProblemBatch, path: str) -> None:
    problems.validate()
    torch.save(asdict(problems), path)


def load_problems(path: str, device: Optional[torch.device] = None) -> ProblemBatch:
    try:
        data = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=True,
        )
    except TypeError:
                                                                       
                                                                           
        data = torch.load(path, map_location=device or "cpu")
                                                                        
    aliases = {
        "agent_capacity": "vehicle_capacity",
        "agent_fixed_cost": "vehicle_fixed_cost",
        "agent_variable_cost": "vehicle_variable_cost",
    }
    for old_name, new_name in aliases.items():
        if old_name in data and new_name not in data:
            data[new_name] = data.pop(old_name)
    problems = ProblemBatch(**data)
    problems.validate()
    return problems


def augment_eight_fold(problems: ProblemBatch) -> ProblemBatch:
    def augment_xy(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :, :1], xy[:, :, 1:]
        variants = (
            torch.cat((x, y), dim=-1),
            torch.cat((1 - x, y), dim=-1),
            torch.cat((x, 1 - y), dim=-1),
            torch.cat((1 - x, 1 - y), dim=-1),
            torch.cat((y, x), dim=-1),
            torch.cat((1 - y, x), dim=-1),
            torch.cat((y, 1 - x), dim=-1),
            torch.cat((1 - y, 1 - x), dim=-1),
        )
        return torch.cat(variants, dim=0)

    return ProblemBatch(
        depot_xy=augment_xy(problems.depot_xy),
        node_xy=augment_xy(problems.node_xy),
        node_demand=problems.node_demand.repeat(8, 1),
        vehicle_capacity=problems.vehicle_capacity.repeat(8, 1),
        vehicle_fixed_cost=problems.vehicle_fixed_cost.repeat(8, 1),
        vehicle_variable_cost=problems.vehicle_variable_cost.repeat(8, 1),
        distance_matrix=(
            None
            if problems.distance_matrix is None
            else problems.distance_matrix.repeat(8, 1, 1)
        ),
    )
