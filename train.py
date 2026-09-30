import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from fsmvrp.model import FSMVRPModel
from fsmvrp.runner import (
    ModelConfig,
    capture_random_state,
    load_checkpoint,
    make_training_batch,
    reinforce_loss,
    restore_random_state,
    rollout,
    save_checkpoint,
    seed_everything,
)


class RolloutTrainingModule(nn.Module):
    """Keep a complete sequential rollout inside one DDP forward call."""

    def __init__(self, policy: FSMVRPModel):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        problems,
        rollout_size: int,
        prune_dominated_types: bool,
        activation_checkpointing: bool,
        compute_entropy: bool,
    ):
        return rollout(
            self.policy,
            problems,
            rollout_size,
            "sampling",
            prune_dominated_types=prune_dominated_types,
            checkpoint_decoder=activation_checkpointing,
            compute_entropy=compute_entropy,
        )


def initialise_process_group(device_name: str) -> Tuple[torch.device, int, int]:
    """Initialise torchrun workers and return device, rank, and world size."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return torch.device(device_name), 0, 1

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requested but CUDA is unavailable.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif device_name == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("Distributed training supports --device cuda or --device cpu.")

    dist.init_process_group(backend=backend)
    return device, dist.get_rank(), dist.get_world_size()


def gather_random_states(
    device: torch.device, world_size: int
) -> Optional[List[dict]]:
    if world_size == 1:
        return None
    states: List[Optional[dict]] = [None] * world_size
    dist.all_gather_object(states, capture_random_state(device))
    return states                              


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the revised FSMVRP solver.")
    parser.add_argument("--problem-size", type=int, default=20)
    parser.add_argument(
        "--problem-sizes",
        help="Optional comma-separated sizes sampled batch by batch.",
    )
    parser.add_argument("--min-vehicle-types", type=int, default=3)
    parser.add_argument("--max-vehicle-types", type=int, default=6)
    parser.add_argument(
        "--vehicle-type-options",
        help="Optional comma-separated vehicle-type counts sampled batch by batch.",
    )
    parser.add_argument(
        "--training-distribution",
        choices=("original", "fsmfd", "fsmfd_v2", "fsmfd_mix"),
        default="original",
    )
    parser.add_argument("--rollout-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--episodes-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/fsmvrp_n20"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--lr-milestones",
        default="500,1000,1500",
        help="Comma-separated epochs after which the learning rate is decayed.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights only and start a fresh optimiser and schedule.",
    )
    parser.add_argument("--prune-dominated-types", action="store_true")
    parser.add_argument("--ensure-nondominated", action="store_true")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Recompute decoder activations during backward to reduce training memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_size < 2:
        raise ValueError("Training requires rollout-size >= 2 for the leave-one-out baseline.")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")
    problem_sizes = (
        [int(value) for value in args.problem_sizes.split(",") if value]
        if args.problem_sizes
        else [args.problem_size]
    )
    if not problem_sizes or any(value < 1 for value in problem_sizes):
        raise ValueError("problem sizes must be positive.")
    vehicle_type_options = (
        tuple(int(value) for value in args.vehicle_type_options.split(",") if value)
        if args.vehicle_type_options
        else None
    )
    if vehicle_type_options is not None and (
        not vehicle_type_options or any(value < 1 for value in vehicle_type_options)
    ):
        raise ValueError("vehicle type options must be positive.")
    device, rank, world_size = initialise_process_group(args.device)
    is_primary = rank == 0
    if args.batch_size % world_size != 0:
        raise ValueError(
            "Global batch-size must be divisible by the distributed world size."
        )
    if args.episodes_per_epoch % world_size != 0:
        raise ValueError(
            "episodes-per-epoch must be divisible by the distributed world size."
        )
    local_batch_size = args.batch_size // world_size
    local_episodes_per_epoch = args.episodes_per_epoch // world_size
    seed_everything(args.seed)
    model_config = ModelConfig(
        use_residual_attention=False,
        use_residual_stats=True,
        use_cost_aware_logits=True,
        use_vehicle_context=True,
        use_hierarchical_fleet_policy=False,
        use_completion_potential=False,
    )
    checkpoint = None
    checkpoint_path = args.resume or args.init_checkpoint
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(str(checkpoint_path), device)
        saved_model_config = asdict(
            ModelConfig(**checkpoint.get("model_config", {}))
        )
        if saved_model_config != asdict(model_config):
            raise ValueError(
                "Checkpoint model configuration differs from the CLI flags."
            )

    model = FSMVRPModel(**asdict(model_config)).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    training_module: nn.Module = RolloutTrainingModule(model)
    if world_size > 1:
        training_module = DistributedDataParallel(
            training_module,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.Adam(
        training_module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    milestones = [int(value) for value in args.lr_milestones.split(",") if value]
    if any(value <= 0 for value in milestones):
        raise ValueError("lr-milestones must contain positive epoch numbers.")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    if is_primary:
        args.output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    run_config = vars(args).copy()
    run_config["output"] = str(run_config["output"])
    run_config["resume"] = None if args.resume is None else str(args.resume)
    run_config["init_checkpoint"] = (
        None if args.init_checkpoint is None else str(args.init_checkpoint)
    )
    run_config["resolved_problem_sizes"] = problem_sizes
    run_config["resolved_vehicle_type_options"] = vehicle_type_options
    run_config["world_size"] = world_size
    run_config["global_batch_size"] = args.batch_size
    run_config["local_batch_size"] = local_batch_size

    start_epoch = 1
    elapsed_before_resume = 0.0
    if args.resume is not None:
        assert checkpoint is not None
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_before_resume = float(
            checkpoint.get("run_config", {}).get("total_train_seconds", 0.0)
        )
        distributed_states = checkpoint.get("distributed_random_states")
        if distributed_states is not None:
            if len(distributed_states) != world_size:
                raise ValueError(
                    "Resume a distributed checkpoint with the same number of processes."
                )
            restore_random_state(distributed_states[rank], device)
        elif world_size == 1:
            restore_random_state(checkpoint, device)
        else:
                                                                         
                                                                           
                                                                         
            seed_everything(args.seed + rank + start_epoch * 1_000_003)
            if is_primary:
                print(
                    "warning: resuming a legacy single-GPU checkpoint with DDP; "
                    "per-rank RNG streams were re-seeded",
                    flush=True,
                )
    elif world_size > 1:
                                                                           
                                                                               
        seed_everything(args.seed + rank)

    metrics_path = args.output / "metrics.csv"
    metrics_mode = "a" if start_epoch > 1 else "w"
    write_header = metrics_mode == "w" or not metrics_path.exists()
    metrics_file = None
    metrics_writer = None
    if is_primary:
        metrics_file = metrics_path.open(metrics_mode, newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "epoch",
                "mean_best_cost",
                "mean_loss",
                "learning_rate",
                "epoch_seconds",
                "total_train_seconds",
                "peak_gpu_memory_mb",
            ),
        )
        if write_header:
            metrics_writer.writeheader()
            metrics_file.flush()
        print(
            "training world_size={} global_batch={} local_batch={} device={}".format(
                world_size, args.batch_size, local_batch_size, device
            ),
            flush=True,
        )

    train_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        training_module.train()
        episode = 0
        cost_sum = 0.0
        loss_sum = 0.0
        batch_count = 0
                                                                         
                                                                     
        epoch_rng = random.Random(args.seed + epoch * 1_000_003)
        epoch_problem_sizes = list(problem_sizes)
        epoch_rng.shuffle(epoch_problem_sizes)
        epoch_vehicle_types = (
            list(vehicle_type_options) if vehicle_type_options else None
        )
        if epoch_vehicle_types is not None:
            epoch_rng.shuffle(epoch_vehicle_types)
        while episode < local_episodes_per_epoch:
            batch_size = min(
                local_batch_size, local_episodes_per_epoch - episode
            )
            problem_size = epoch_problem_sizes[
                batch_count % len(epoch_problem_sizes)
            ]
            selected_vehicle_types = (
                None
                if epoch_vehicle_types is None
                else (
                    epoch_vehicle_types[
                        batch_count % len(epoch_vehicle_types)
                    ],
                )
            )
            problems = make_training_batch(
                batch_size,
                problem_size,
                args.min_vehicle_types,
                args.max_vehicle_types,
                device,
                args.ensure_nondominated,
                problem_distribution=args.training_distribution,
                vehicle_type_options=selected_vehicle_types,
            )
            reward, log_probability, entropy = training_module(
                problems,
                args.rollout_size,
                args.prune_dominated_types,
                args.activation_checkpointing,
                args.entropy_coefficient != 0.0,
            )
            loss, _ = reinforce_loss(
                reward,
                log_probability,
                entropy,
                args.entropy_coefficient,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                training_module.parameters(), args.max_grad_norm
            )
            optimizer.step()

            cost_sum += float((-reward.max(dim=1).values.mean()).item())
            loss_sum += float(loss.item())
            batch_count += 1
            episode += batch_size
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_gpu_memory_mb = 0.0
        epoch_seconds = time.perf_counter() - epoch_start
        total_train_seconds = elapsed_before_resume + time.perf_counter() - train_start
        sums = torch.tensor(
            [cost_sum, loss_sum, float(batch_count)],
            device=device,
            dtype=torch.float64,
        )
        maxima = torch.tensor(
            [epoch_seconds, total_train_seconds, peak_gpu_memory_mb],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        mean_cost = float((sums[0] / sums[2]).item())
        mean_loss = float((sums[1] / sums[2]).item())
        epoch_seconds, total_train_seconds, peak_gpu_memory_mb = (
            float(value) for value in maxima.tolist()
        )
        if is_primary:
            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "mean_best_cost": mean_cost,
                    "mean_loss": mean_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": epoch_seconds,
                    "total_train_seconds": total_train_seconds,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                }
            )
            metrics_file.flush()
            print(
                "epoch={:04d} cost={:.6f} loss={:.6f} lr={:.3e} seconds={:.2f} peak_mb={:.1f}".format(
                    epoch,
                    mean_cost,
                    mean_loss,
                    optimizer.param_groups[0]["lr"],
                    epoch_seconds,
                    peak_gpu_memory_mb,
                ),
                flush=True,
            )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            run_config["total_train_seconds"] = total_train_seconds
            distributed_random_states = gather_random_states(device, world_size)
            if is_primary:
                save_checkpoint(
                    args.output / "checkpoint-{}.pt".format(epoch),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    model_config,
                    run_config,
                    distributed_random_states=distributed_random_states,
                )
            if world_size > 1:
                dist.barrier()

    if metrics_file is not None:
        metrics_file.close()
    total_train_seconds = elapsed_before_resume + time.perf_counter() - train_start
    if world_size > 1:
        total_tensor = torch.tensor(
            total_train_seconds, device=device, dtype=torch.float64
        )
        dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
        total_train_seconds = float(total_tensor.item())
    if is_primary:
        with (args.output / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "completed_epochs": args.epochs,
                    "start_epoch": start_epoch,
                    "total_train_seconds": total_train_seconds,
                    "device": str(device),
                    "world_size": world_size,
                    "global_batch_size": args.batch_size,
                    "local_batch_size": local_batch_size,
                },
                file,
                indent=2,
                sort_keys=True,
            )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
