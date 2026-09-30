import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .env import StepState
from .problem import ProblemBatch


def instance_scales(
    problems: ProblemBatch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coordinate, demand, and objective scales for each instance."""
    all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
    coordinate_range = all_xy.max(dim=1).values - all_xy.min(dim=1).values
    coordinate_scale = coordinate_range.max(dim=1).values.clamp_min(1e-8)
    demand_scale = problems.vehicle_capacity.max(dim=1).values.clamp_min(1e-8)
    objective_scale = (
        problems.vehicle_fixed_cost.mean(dim=1)
        + problems.vehicle_variable_cost.mean(dim=1) * coordinate_scale
    ).clamp_min(1e-8)
    return coordinate_scale, demand_scale, objective_scale


def masked_softmax(
    score: torch.Tensor, valid: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Numerically safe masked softmax, including fully masked query rows."""
    valid = valid.expand_as(score)
    masked_score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
    weight = F.softmax(masked_score, dim=dim) * valid.to(score.dtype)
    denominator = weight.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        weight / denominator.clamp_min(torch.finfo(score.dtype).eps),
        torch.zeros_like(weight),
    )


def factorized_joint_distribution(
    node_logits: torch.Tensor,
    type_logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return p(vehicle type) * p(node | vehicle type).

    A flat softmax assigns more marginal probability to a vehicle type merely
    when that type has more feasible customer actions.  FSMVRP capacities make
    those action counts systematically different across types.  Normalising
    the type and node decisions separately removes that artefact while still
    producing one joint vehicle-node action at every environment step.
    """
    if node_logits.shape != valid.shape:
        raise ValueError("node_logits and valid must have the same shape.")
    if type_logits.shape != node_logits.shape[:-1]:
        raise ValueError("type_logits must match node_logits without its last axis.")
    node_probability = masked_softmax(node_logits, valid, dim=-1)
    type_valid = valid.any(dim=-1)
    type_probability = masked_softmax(type_logits, type_valid, dim=-1)
    return type_probability.unsqueeze(-1) * node_probability


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    return_weights: bool = False,
) -> torch.Tensor:
    """Standard attention: softmax is applied over keys for every query."""
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    weight = F.softmax(score, dim=-1) if valid is None else masked_softmax(score, valid, dim=-1)
    output = torch.matmul(weight, value)
    return (output, weight) if return_weights else output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        inner_dim = head_num * qkv_dim
        self.query = nn.Linear(embedding_dim, inner_dim, bias=False)
        self.key = nn.Linear(embedding_dim, inner_dim, bias=False)
        self.value = nn.Linear(embedding_dim, inner_dim, bias=False)
        self.output = nn.Linear(inner_dim, embedding_dim)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape[:-1] + (self.head_num, self.qkv_dim)
        tensor = tensor.reshape(shape)
        return tensor.transpose(-3, -2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(-3, -2).contiguous()
        return tensor.reshape(tensor.shape[:-2] + (self.head_num * self.qkv_dim,))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split(self.query(query))
        k = self._split(self.key(key))
        v = self._split(self.value(value))
        if valid is not None:
            valid = valid.unsqueeze(-3)
        attended = scaled_dot_product_attention(q, k, v, valid)
        return self.output(self._merge(attended))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, head_num, qkv_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        nodes = self.norm_1(nodes + self.attention(nodes, nodes, nodes))
        return self.norm_2(nodes + self.feed_forward(nodes))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        hidden_dim: int,
        layer_num: int,
    ):
        super().__init__()
        self.depot_embedding = nn.Linear(2, embedding_dim)
        self.customer_embedding = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, head_num, qkv_dim, hidden_dim)
                for _ in range(layer_num)
            ]
        )

    def forward(self, problems: ProblemBatch) -> torch.Tensor:
        all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
        xy_min = all_xy.min(dim=1, keepdim=True).values
        coordinate_scale, demand_scale, _ = instance_scales(problems)
        normalized_xy = (all_xy - xy_min) / coordinate_scale[:, None, None]
        depot = self.depot_embedding(normalized_xy[:, :1])
        customer_input = torch.cat(
            (
                normalized_xy[:, 1:],
                problems.node_demand[:, :, None] / demand_scale[:, None, None],
            ),
            dim=-1,
        )
        customer = self.customer_embedding(customer_input)
        nodes = torch.cat((depot, customer), dim=1)
        for layer in self.layers:
            nodes = layer(nodes)
        return nodes


class VehicleContextLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, head_num, qkv_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        context = self.norm_1(
            context + self.attention(context, context, context)
        )
        return self.norm_2(context + self.feed_forward(context))


class DynamicJointDecoder(nn.Module):
    """Dynamic residual-demand and objective-aware joint decoder."""

    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        hidden_dim: int,
        context_layer_num: int,
        logit_clipping: float,
        use_residual_attention: bool,
        use_residual_stats: bool,
        use_cost_aware_logits: bool,
        use_vehicle_context: bool,
        use_hierarchical_fleet_policy: bool,
        use_completion_potential: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logit_clipping = logit_clipping
        self.use_residual_attention = use_residual_attention
        self.use_residual_stats = use_residual_stats
        self.use_cost_aware_logits = use_cost_aware_logits
        self.use_vehicle_context = use_vehicle_context
        self.use_hierarchical_fleet_policy = use_hierarchical_fleet_policy
        self.use_completion_potential = use_completion_potential

        if use_residual_attention:
            self.residual_query = nn.Linear(
                embedding_dim + 4, embedding_dim, bias=False
            )
            self.residual_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.residual_value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.residual_query = None
            self.residual_key = None
            self.residual_value = None
        self.stats_embedding = nn.Linear(4, embedding_dim, bias=False)
        self.residual_gate = nn.Linear(2 * embedding_dim + 8, embedding_dim)
        self.residual_norm = nn.LayerNorm(embedding_dim)

        self.context_embedding = nn.Linear(2 * embedding_dim + 4, embedding_dim)
        self.context_layers = nn.ModuleList(
            [
                VehicleContextLayer(embedding_dim, head_num, qkv_dim, hidden_dim)
                for _ in range(context_layer_num)
            ]
        )
        self.node_attention = MultiHeadAttention(embedding_dim, head_num, qkv_dim)
        self.decision_norm_1 = nn.LayerNorm(embedding_dim)
        self.decision_ff = FeedForward(embedding_dim, hidden_dim)
        self.decision_norm_2 = nn.LayerNorm(embedding_dim)
        self.final_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.final_key = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.edge_score = nn.Sequential(
            nn.Linear(5, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
                                                                               
                                                 
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)

        if use_completion_potential:
                                                                           
                                                                           
                                                                           
                                                                     
            self.completion_weight = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("completion_weight", None)

        if use_hierarchical_fleet_policy:
                                                                          
                                                                            
                                                      
            self.type_score = nn.Sequential(
                nn.Linear(embedding_dim + 4, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )
        else:
            self.type_score = None

    @staticmethod
    def _gather_nodes(encoded_nodes: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch_size, rollout_size, type_num = index.shape
        embedding_dim = encoded_nodes.size(-1)
        expanded = encoded_nodes[:, None, :, :].expand(
            batch_size, rollout_size, -1, embedding_dim
        )
        return expanded.gather(
            2, index[:, :, :, None].expand(-1, -1, -1, embedding_dim)
        )

    def _residual_representation(
        self,
        base_state: torch.Tensor,
        encoded_nodes: torch.Tensor,
        problems: ProblemBatch,
        state: StepState,
    ) -> torch.Tensor:
        batch_size, rollout_size, type_num, _ = base_state.shape
        residual_valid = ~state.action_mask
        residual_valid = residual_valid.clone()
        residual_valid[:, :, :, 0] = False
        if self.use_residual_attention:
            if (
                self.residual_query is None
                or self.residual_key is None
                or self.residual_value is None
            ):
                raise RuntimeError("Residual-attention layers are unavailable.")
            query = self.residual_query(base_state)
            key = self.residual_key(encoded_nodes)[:, None, :, :]
            value = self.residual_value(encoded_nodes)[:, None, :, :]
            score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                self.embedding_dim
            )
            weight = masked_softmax(score, residual_valid, dim=-1)
            residual = torch.matmul(
                weight, value.expand(-1, rollout_size, -1, -1)
            )
        else:
                                                                            
                                                                          
            residual = base_state.new_zeros(
                batch_size, rollout_size, type_num, self.embedding_dim
            )

        unvisited = ~state.visited[:, :, 1:]
        remaining_count = unvisited.sum(dim=-1, keepdim=True).float()
        demand = problems.node_demand[:, None, :]
        remaining_demand = (unvisited.float() * demand).sum(dim=-1, keepdim=True)
        feasible = residual_valid[:, :, :, 1:]
        feasible_count = feasible.sum(dim=-1, keepdim=True).float()
        feasible_demand = (
            feasible.float() * problems.node_demand[:, None, None, :]
        ).sum(dim=-1, keepdim=True)
        capacity = problems.vehicle_capacity[:, None, :, None].clamp_min(1e-8)
        expanded_count = remaining_count[:, :, None, :].expand(
            -1, -1, type_num, -1
        )
        stats = torch.cat(
            (
                expanded_count / max(problems.problem_size, 1),
                remaining_demand[:, :, None, :].expand(-1, -1, type_num, -1)
                / capacity,
                feasible_count / expanded_count.clamp_min(1.0),
                feasible_demand / feasible_count.clamp_min(1.0) / capacity,
            ),
            dim=-1,
        )
        effective_stats = stats if self.use_residual_stats else torch.zeros_like(stats)
        stats_embedding = self.stats_embedding(effective_stats)
        residual = self.residual_norm(residual + stats_embedding)
        gate = torch.sigmoid(
            self.residual_gate(
                torch.cat((base_state, residual, effective_stats), dim=-1)
            )
        )
        return gate * residual

    def _edge_bias(
        self, problems: ProblemBatch, state: StepState
    ) -> torch.Tensor:
        batch_size, rollout_size, type_num = state.current_node.shape
        graph_size = problems.problem_size + 1
        all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
        if problems.distance_matrix is None:
            expanded_xy = all_xy[:, None, :, :].expand(
                batch_size, rollout_size, graph_size, 2
            )
            current_xy = expanded_xy.gather(
                2, state.current_node[:, :, :, None].expand(-1, -1, -1, 2)
            )
            distance = (
                current_xy[:, :, :, None, :] - expanded_xy[:, :, None, :, :]
            ).pow(2).sum(dim=-1).sqrt()
        else:
            matrix = problems.distance_matrix[:, None, None, :, :].expand(
                batch_size, rollout_size, type_num, graph_size, graph_size
            )
            distance = matrix.gather(
                3,
                state.current_node[:, :, :, None, None].expand(
                    -1, -1, -1, 1, graph_size
                ),
            ).squeeze(3)
        target_demand = torch.cat(
            (torch.zeros_like(problems.node_demand[:, :1]), problems.node_demand),
            dim=1,
        )[:, None, None, :].expand(batch_size, rollout_size, type_num, graph_size)
        capacity = problems.vehicle_capacity[:, None, :, None].clamp_min(1e-8)
        variable = problems.vehicle_variable_cost[:, None, :, None]
        coordinate_scale, _, objective_scale = instance_scales(problems)
        normalized_distance = distance / coordinate_scale[:, None, None, None]
        normalized_travel_cost = (
            variable * distance / objective_scale[:, None, None, None]
        )
        customer_target = torch.ones(
            graph_size, device=distance.device, dtype=distance.dtype
        )
        customer_target[0] = 0
        fixed_departure = (
            state.at_the_depot[:, :, :, None].to(distance.dtype)
            * (
                problems.vehicle_fixed_cost[:, None, :, None]
                / objective_scale[:, None, None, None]
            )
            * customer_target[None, None, None, :]
        )
        load_after = (state.load[:, :, :, None] - target_demand).clamp_min(0) / capacity
        features = torch.stack(
            (
                normalized_distance,
                normalized_travel_cost,
                target_demand / capacity,
                load_after,
                fixed_departure,
            ),
            dim=-1,
        )
        return self.edge_score(features).squeeze(-1)

    def _completion_marginal_cost(
        self, problems: ProblemBatch, state: StepState
    ) -> torch.Tensor:
        """Exact normalised change in route cost if an action is appended.

        Closing every currently open route defines a partial-solution cost.
        Appending node j to the open route of type k changes that quantity by

          c_k [d(i_k, j) + d(j, 0) - d(i_k, 0)]
          + f_k 1[i_k = 0, j != 0].

        This is specific to the fixed-plus-variable FSMVRP objective.  It is
        not a generic distance heuristic and remains exact for an asymmetric
        distance matrix.
        """
        batch_size, rollout_size, type_num = state.current_node.shape
        graph_size = problems.problem_size + 1
        all_xy = torch.cat((problems.depot_xy, problems.node_xy), dim=1)
        if problems.distance_matrix is None:
            expanded_xy = all_xy[:, None, :, :].expand(
                batch_size, rollout_size, graph_size, 2
            )
            current_xy = expanded_xy.gather(
                2, state.current_node[:, :, :, None].expand(-1, -1, -1, 2)
            )
            distance = (
                current_xy[:, :, :, None, :] - expanded_xy[:, :, None, :, :]
            ).pow(2).sum(dim=-1).sqrt()
            depot = problems.depot_xy[:, None, :, :]
            current_return = (current_xy - depot).pow(2).sum(dim=-1).sqrt()
            target_return = (
                all_xy - problems.depot_xy
            ).pow(2).sum(dim=-1).sqrt()
        else:
            matrix = problems.distance_matrix
            expanded = matrix[:, None, None, :, :].expand(
                batch_size, rollout_size, type_num, graph_size, graph_size
            )
            distance = expanded.gather(
                3,
                state.current_node[:, :, :, None, None].expand(
                    -1, -1, -1, 1, graph_size
                ),
            ).squeeze(3)
            return_column = matrix[:, :, 0]
            current_return = return_column[:, None, :].expand(
                batch_size, rollout_size, graph_size
            ).gather(2, state.current_node)
            target_return = return_column

        route_extension = (
            distance
            + target_return[:, None, None, :]
            - current_return[:, :, :, None]
        )
        variable_cost = problems.vehicle_variable_cost[:, None, :, None]
        customer_target = torch.ones(
            graph_size, device=distance.device, dtype=distance.dtype
        )
        customer_target[0] = 0
        fixed_departure = (
            state.at_the_depot[:, :, :, None].to(distance.dtype)
            * problems.vehicle_fixed_cost[:, None, :, None]
            * customer_target[None, None, None, :]
        )
        _, _, objective_scale = instance_scales(problems)
        return (
            variable_cost * route_extension + fixed_departure
        ) / objective_scale[:, None, None, None]

    def _type_logits(
        self,
        decision: torch.Tensor,
        node_logits: torch.Tensor,
        completion_potential: torch.Tensor,
        valid: torch.Tensor,
        state: StepState,
    ) -> torch.Tensor:
        if self.type_score is None:
            raise RuntimeError("The hierarchical fleet policy is disabled.")
        type_valid = valid.any(dim=-1)
        valid_count = valid.sum(dim=-1).clamp_min(1)
        masked_logits = node_logits.masked_fill(~valid, float("-inf"))
                                                                         
                                                    
        evidence = torch.logsumexp(masked_logits, dim=-1) - valid_count.log()
        evidence = torch.where(type_valid, evidence, torch.zeros_like(evidence))
        evidence = evidence / max(self.logit_clipping, 1e-8)

        potential_weight = masked_softmax(-completion_potential, valid, dim=-1)
        potential_summary = (potential_weight * completion_potential).sum(dim=-1)

        feasible_customer = valid[:, :, :, 1:].sum(dim=-1).float()
        remaining_customer = (
            ~state.visited[:, :, 1:]
        ).sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        feasible_fraction = feasible_customer / remaining_customer
        opening = state.at_the_depot.to(decision.dtype)
        features = torch.cat(
            (
                decision,
                evidence[:, :, :, None],
                -potential_summary[:, :, :, None],
                feasible_fraction[:, :, :, None],
                opening[:, :, :, None],
            ),
            dim=-1,
        )
        return self.type_score(features).squeeze(-1)

    def forward(
        self,
        encoded_nodes: torch.Tensor,
        problems: ProblemBatch,
        state: StepState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rollout_size = state.rollout_size
        current = self._gather_nodes(encoded_nodes, state.current_node)
        capacity = problems.vehicle_capacity[:, None, :]
        load_ratio = state.load / capacity.clamp_min(1e-8)
        coordinate_scale, demand_scale, objective_scale = instance_scales(problems)
        capacity_ratio = capacity / demand_scale[:, None, None]
        fixed_feature = (
            state.at_the_depot.float()
            * (
                problems.vehicle_fixed_cost[:, None, :]
                / objective_scale[:, None, None]
            )
        )
        variable_feature = (
            problems.vehicle_variable_cost[:, None, :]
            * coordinate_scale[:, None, None]
            / objective_scale[:, None, None]
        ).expand(-1, rollout_size, -1)
        base_state = torch.cat(
            (
                current,
                load_ratio[:, :, :, None],
                capacity_ratio.expand(-1, rollout_size, -1)[:, :, :, None],
                fixed_feature[:, :, :, None],
                variable_feature[:, :, :, None],
            ),
            dim=-1,
        )
        residual = self._residual_representation(
            base_state, encoded_nodes, problems, state
        )
        context = self.context_embedding(
            torch.cat(
                (
                    current,
                    residual,
                    load_ratio[:, :, :, None],
                    capacity_ratio.expand(-1, rollout_size, -1)[:, :, :, None],
                    fixed_feature[:, :, :, None],
                    variable_feature[:, :, :, None],
                ),
                dim=-1,
            )
        )
        if self.use_vehicle_context:
            for layer in self.context_layers:
                context = layer(context)

        node_values = encoded_nodes[:, None, :, :]
        cross = self.node_attention(
            context,
            node_values,
            node_values,
            valid=~state.action_mask,
        )
        decision = self.decision_norm_1(context + cross)
        decision = self.decision_norm_2(decision + self.decision_ff(decision))
        query = self.final_query(decision)
        key = self.final_key(encoded_nodes)[:, None, :, :]
        compatibility = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.embedding_dim
        )
        if self.use_cost_aware_logits:
            compatibility = compatibility + self._edge_bias(problems, state)
        if self.use_completion_potential:
            completion_potential = self._completion_marginal_cost(problems, state)
            compatibility = compatibility - F.softplus(
                self.completion_weight
            ) * completion_potential
        else:
            completion_potential = torch.zeros_like(compatibility)
        logits = self.logit_clipping * torch.tanh(compatibility)

        batch_size = problems.batch_size
        valid = ~state.action_mask
        flat_valid = valid.reshape(batch_size, rollout_size, -1)
        if self.use_hierarchical_fleet_policy:
            type_logits = self._type_logits(
                decision,
                logits,
                completion_potential,
                valid,
                state,
            )
            joint_probability = factorized_joint_distribution(
                logits, type_logits, valid
            )
            probabilities = joint_probability.reshape(batch_size, rollout_size, -1)
            selection_logits = probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log()
        else:
            flat_logits = logits.reshape(batch_size, rollout_size, -1)
            probabilities = masked_softmax(flat_logits, flat_valid, dim=-1)
            selection_logits = flat_logits
        return probabilities, selection_logits.masked_fill(
            ~flat_valid, torch.finfo(selection_logits.dtype).min
        )


class FSMVRPModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        head_num: int = 8,
        qkv_dim: int = 16,
        hidden_dim: int = 512,
        context_layer_num: int = 1,
        logit_clipping: float = 10.0,
        use_residual_attention: bool = False,
        use_residual_stats: bool = True,
        use_cost_aware_logits: bool = True,
        use_vehicle_context: bool = True,
        use_hierarchical_fleet_policy: bool = False,
        use_completion_potential: bool = False,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            embedding_dim,
            head_num,
            qkv_dim,
            hidden_dim,
            encoder_layer_num,
        )
        self.decoder = DynamicJointDecoder(
            embedding_dim,
            head_num,
            qkv_dim,
            hidden_dim,
            context_layer_num,
            logit_clipping,
            use_residual_attention,
            use_residual_stats,
            use_cost_aware_logits,
            use_vehicle_context,
            use_hierarchical_fleet_policy,
            use_completion_potential,
        )
        self.encoded_nodes: Optional[torch.Tensor] = None

    def pre_forward(self, problems: ProblemBatch) -> None:
        self.encoded_nodes = self.encoder(problems)

    def forward(
        self, problems: ProblemBatch, state: StepState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoded_nodes is None:
            raise RuntimeError("Call pre_forward once after resetting the environment.")
        return self.decoder(self.encoded_nodes, problems, state)

    def select_action(
        self,
        problems: ProblemBatch,
        state: StepState,
        decode_type: str = "sampling",
        checkpoint_decoder: bool = False,
        compute_entropy: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if checkpoint_decoder and self.training and torch.is_grad_enabled():
            if self.encoded_nodes is None:
                raise RuntimeError("Call pre_forward once after resetting the environment.")

            def decode(encoded_nodes: torch.Tensor):
                return self.decoder(encoded_nodes, problems, state)

                                                                           
                                                                              
                                                                            
                                                                      
            major_version = int(torch.__version__.split(".", 1)[0])
            if major_version >= 2:
                probabilities, logits = checkpoint(
                    decode, self.encoded_nodes, use_reentrant=False
                )
            else:
                probabilities, logits = checkpoint(decode, self.encoded_nodes)
        else:
            probabilities, logits = self(problems, state)
        distribution = torch.distributions.Categorical(probs=probabilities)
        if decode_type == "sampling":
            action = distribution.sample()
        elif decode_type == "greedy":
            action = logits.argmax(dim=-1)
        else:
            raise ValueError("decode_type must be 'sampling' or 'greedy'.")
        selected_probability = probabilities.gather(-1, action[:, :, None]).squeeze(-1)
        entropy = (
            distribution.entropy()
            if compute_entropy
            else torch.zeros_like(selected_probability)
        )
        return action, selected_probability, entropy
