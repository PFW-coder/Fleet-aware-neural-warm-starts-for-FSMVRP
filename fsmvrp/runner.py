import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .env import FSMVRPEnv
from .model import FSMVRPModel
from .problem import (
    ProblemBatch,
    augment_eight_fold,
    generate_fsmfd_problems,
    generate_fsmfd_v2_problems,
    generate_problems,
)
from .solution import (
    ConstructedSolution,
    decode_action_sequence,
    validate_and_cost_solution,
)


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    encoder_layer_num: int = 6
    head_num: int = 8
    qkv_dim: int = 16
    hidden_dim: int = 512
    context_layer_num: int = 1
    logit_clipping: float = 10.0
    use_residual_attention: bool = False
    use_residual_stats: bool = True
    use_cost_aware_logits: bool = True
    use_vehicle_context: bool = True
    use_hierarchical_fleet_policy: bool = False
    use_completion_potential: bool = False


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rollout_with_actions(
    model: FSMVRPModel,
    problems: ProblemBatch,
    rollout_size: int,
    decode_type: str,
    prune_dominated_types: bool = False,
    checkpoint_decoder: bool = False,
    compute_entropy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    env = FSMVRPEnv(
        rollout_size=rollout_size,
        prune_dominated_types=prune_dominated_types,
    )
    _, state = env.reset(problems)
    model.pre_forward(problems)
    log_probabilities = []
    entropies = []
    reward = None
    max_steps = 2 * problems.problem_size + problems.vehicle_type_num
    for _ in range(max_steps):
        action, probability, entropy = model.select_action(
            problems,
            state,
            decode_type=decode_type,
            checkpoint_decoder=checkpoint_decoder,
            compute_entropy=compute_entropy,
        )
        log_probabilities.append(probability.clamp_min(1e-12).log())
        entropies.append(entropy)
        state, reward, done = env.step(action)
        if done:
            break
    else:
        raise RuntimeError("Rollout exceeded the theoretical construction-step bound.")
    return (
        reward,
        torch.stack(log_probabilities, dim=-1).sum(dim=-1),
        torch.stack(entropies, dim=-1).mean(dim=-1),
        torch.stack(env.action_history, dim=-1),
    )


def rollout(
    model: FSMVRPModel,
    problems: ProblemBatch,
    rollout_size: int,
    decode_type: str,
    prune_dominated_types: bool = False,
    checkpoint_decoder: bool = False,
    compute_entropy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reward, log_probability, entropy, _ = _rollout_with_actions(
        model,
        problems,
        rollout_size,
        decode_type,
        prune_dominated_types=prune_dominated_types,
        checkpoint_decoder=checkpoint_decoder,
        compute_entropy=compute_entropy,
    )
    return reward, log_probability, entropy


def reinforce_loss(
    reward: torch.Tensor,
    log_probability: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coefficient: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rollout_size = reward.size(1)
    if rollout_size > 1:
        baseline = (reward.sum(dim=1, keepdim=True) - reward) / (rollout_size - 1)
    else:
        baseline = reward.mean().expand_as(reward)
    advantage = reward - baseline
    policy_loss = -(advantage.detach() * log_probability).mean()
    loss = policy_loss - entropy_coefficient * entropy.mean()
    return loss, advantage


@torch.no_grad()
def evaluate_costs(
    model: FSMVRPModel,
    problems: ProblemBatch,
    rollout_size: int = 1,
    decode_type: str = "greedy",
    augment: bool = False,
    prune_dominated_types: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one unaugmented and one best-augmentation cost per instance."""
    model.eval()
    base_batch_size = problems.batch_size
    evaluated = augment_eight_fold(problems) if augment else problems
    reward, _, _ = rollout(
        model,
        evaluated,
        rollout_size,
        decode_type,
        prune_dominated_types=prune_dominated_types,
    )
    cost = -reward
    best_rollout = cost.min(dim=1).values
    no_augmentation = best_rollout[:base_batch_size]
    if augment:
        best_augmentation = best_rollout.reshape(8, base_batch_size).min(dim=0).values
    else:
        best_augmentation = best_rollout
    return no_augmentation, best_augmentation


@torch.no_grad()
def evaluate_solution_plans(
    model: FSMVRPModel,
    problems: ProblemBatch,
    rollout_size: int = 1,
    decode_type: str = "greedy",
    augment: bool = False,
    prune_dominated_types: bool = False,
    augmentation_chunk_size: int = 8,
) -> List[ConstructedSolution]:
    """Return the best explicit route plan for every problem instance.

    When eight-fold augmentation is enabled, ``augmentation_chunk_size``
    controls how many transformed copies are encoded simultaneously.  A value
    of eight preserves the original batched path.  Smaller values evaluate the
    same eight transformations in several passes, reducing peak encoder memory
    for large instances without changing the candidate set.
    """
    if augmentation_chunk_size < 1 or augmentation_chunk_size > 8:
        raise ValueError("augmentation_chunk_size must be between 1 and 8.")
    model.eval()
    base_batch_size = problems.batch_size
    augmentation_count = 8 if augment else 1
    evaluated = augment_eight_fold(problems) if augment else problems

    chunk_size = min(augmentation_chunk_size, augmentation_count)
    best_costs = torch.full(
        (base_batch_size,),
        float("inf"),
        device=problems.node_xy.device,
        dtype=problems.node_xy.dtype,
    )
    best_augmentations = [-1] * base_batch_size
    best_rollouts = [-1] * base_batch_size
    best_actions: List[Optional[List[int]]] = [None] * base_batch_size

    for augmentation_start in range(0, augmentation_count, chunk_size):
        augmentation_stop = min(
            augmentation_start + chunk_size, augmentation_count
        )
        evaluated_start = augmentation_start * base_batch_size
        evaluated_stop = augmentation_stop * base_batch_size
        chunk = evaluated.slice(evaluated_start, evaluated_stop)
        reward, _, _, actions = _rollout_with_actions(
            model,
            chunk,
            rollout_size,
            decode_type,
            prune_dominated_types=prune_dominated_types,
            compute_entropy=False,
        )
        chunk_augmentations = augmentation_stop - augmentation_start
        grouped_costs = (-reward).reshape(
            chunk_augmentations, base_batch_size, rollout_size
        ).permute(1, 0, 2)
        chunk_best_costs, chunk_best_indices = grouped_costs.reshape(
            base_batch_size, -1
        ).min(dim=1)

        for instance_index in range(base_batch_size):
            if chunk_best_costs[instance_index] >= best_costs[instance_index]:
                continue
            flat_index = int(chunk_best_indices[instance_index].item())
            local_augmentation, rollout_index = divmod(flat_index, rollout_size)
            evaluated_index = local_augmentation * base_batch_size + instance_index
            best_costs[instance_index] = chunk_best_costs[instance_index]
            best_augmentations[instance_index] = (
                augmentation_start + local_augmentation
            )
            best_rollouts[instance_index] = rollout_index
            best_actions[instance_index] = (
                actions[evaluated_index, rollout_index].detach().cpu().tolist()
            )

    solutions = []
    for instance_index in range(base_batch_size):
        action_sequence = best_actions[instance_index]
        if action_sequence is None:
            raise RuntimeError("No augmented rollout was evaluated.")
        augmentation_index = best_augmentations[instance_index]
        rollout_index = best_rollouts[instance_index]
        route_plan = decode_action_sequence(
            action_sequence,
            problems.problem_size,
            problems.vehicle_type_num,
        )
        reconstructed_cost = validate_and_cost_solution(
            problems, instance_index, route_plan
        )
        decoded_cost = float(best_costs[instance_index].item())
        tolerance = 1e-4 * max(1.0, abs(decoded_cost))
        if abs(reconstructed_cost - decoded_cost) > tolerance:
            raise RuntimeError(
                "Decoded route cost does not match the environment objective: "
                "{} versus {}.".format(reconstructed_cost, decoded_cost)
            )
        solutions.append(
            ConstructedSolution(
                cost=reconstructed_cost,
                routes=route_plan,
                augmentation_index=augmentation_index,
                rollout_index=rollout_index,
            )
        )
    return solutions


@torch.no_grad()
def evaluate(
    model: FSMVRPModel,
    problems: ProblemBatch,
    rollout_size: int = 1,
    decode_type: str = "greedy",
    augment: bool = False,
    prune_dominated_types: bool = False,
) -> Dict[str, float]:
    no_augmentation, best_augmentation = evaluate_costs(
        model,
        problems,
        rollout_size=rollout_size,
        decode_type=decode_type,
        augment=augment,
        prune_dominated_types=prune_dominated_types,
    )
    return {
        "mean_cost": float(no_augmentation.mean().item()),
        "mean_augmented_cost": float(best_augmentation.mean().item()),
    }


def save_checkpoint(
    path: Path,
    model: FSMVRPModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    model_config: ModelConfig,
    run_config: Dict,
    distributed_random_states: Optional[List[Dict]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
                                                                              
                                                                               
                                                                        
    random_state = (
        distributed_random_states[0]
        if distributed_random_states is not None
        else capture_random_state()
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "model_config": asdict(model_config),
            "run_config": run_config,
            **random_state,
            "distributed_random_states": distributed_random_states,
        },
        str(path),
    )
    with (path.parent / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"model": asdict(model_config), "run": run_config},
            file,
            indent=2,
            sort_keys=True,
        )


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[FSMVRPModel, Dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config", {})
    model = FSMVRPModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load a tensor checkpoint across supported PyTorch versions."""
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def capture_random_state(device: Optional[torch.device] = None) -> Dict:
    """Capture RNG streams for one process.

    A distributed worker stores only its local CUDA generator.  Legacy
    single-process checkpoints keep all visible CUDA generators so their
    schema remains backward compatible.
    """
    if not torch.cuda.is_available():
        cuda_state = None
    elif device is not None and torch.device(device).type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    else:
        cuda_state = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": cuda_state,
    }


def restore_random_state(
    checkpoint: Dict, device: Optional[torch.device] = None
) -> None:
    """Restore RNG streams saved at an epoch boundary, when available."""
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    cuda_state = checkpoint.get("cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        if torch.is_tensor(cuda_state):
            target = device if device is not None else torch.cuda.current_device()
            torch.cuda.set_rng_state(cuda_state.cpu(), target)
        else:
            torch.cuda.set_rng_state_all(cuda_state)


def make_training_batch(
    batch_size: int,
    problem_size: int,
    min_vehicle_types: int,
    max_vehicle_types: int,
    device: torch.device,
    ensure_nondominated: bool,
    problem_distribution: str = "original",
    vehicle_type_options: Optional[Tuple[int, ...]] = None,
) -> ProblemBatch:
    type_num = (
        random.choice(vehicle_type_options)
        if vehicle_type_options
        else random.randint(min_vehicle_types, max_vehicle_types)
    )
    if problem_distribution == "fsmfd":
        return generate_fsmfd_problems(
            batch_size=batch_size,
            problem_size=problem_size,
            vehicle_type_num=type_num,
            device=device,
        )
    if problem_distribution == "fsmfd_v2":
        return generate_fsmfd_v2_problems(
            batch_size=batch_size,
            problem_size=problem_size,
            vehicle_type_num=type_num,
            device=device,
        )
    if problem_distribution == "fsmfd_mix":
        generator = (
            generate_fsmfd_v2_problems
            if random.random() < 0.5
            else generate_fsmfd_problems
        )
        return generator(
            batch_size=batch_size,
            problem_size=problem_size,
            vehicle_type_num=type_num,
            device=device,
        )
    if problem_distribution != "original":
        raise ValueError("Unknown problem distribution: {}".format(problem_distribution))
    return generate_problems(
        batch_size=batch_size,
        problem_size=problem_size,
        vehicle_type_num=type_num,
        device=device,
        ensure_nondominated=ensure_nondominated,
    )
