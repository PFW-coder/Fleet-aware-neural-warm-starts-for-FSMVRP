import argparse
import csv
import json
import time
from pathlib import Path

import torch

from fsmvrp.problem import dominated_vehicle_mask, generate_problems, load_problems
from fsmvrp.runner import evaluate_costs, load_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a revised FSMVRP model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--problem-size", type=int, default=20)
    parser.add_argument("--vehicle-types", type=int, default=6)
    parser.add_argument("--test-instances", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rollout-size", type=int, default=1000)
    parser.add_argument("--decode-type", choices=("greedy", "sampling"), default="sampling")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prune-dominated-types", action="store_true")
    parser.add_argument("--method-name", default="fsmvrp_drl")
    parser.add_argument("--dataset-name")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)
    model, checkpoint = load_model(args.checkpoint, device)

    loaded = load_problems(args.data, device) if args.data else None
    test_instances = args.test_instances
    if test_instances is None:
        test_instances = loaded.batch_size if loaded is not None else 1000
    if loaded is not None and test_instances > loaded.batch_size:
        raise ValueError(
            "test-instances exceeds the number of instances in the data file."
        )
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = Path(args.data).stem if args.data else "generated"

    rows = []
    total = 0
    cost_sum = 0.0
    augmented_sum = 0.0
    evaluation_start = time.perf_counter()
    while total < test_instances:
        batch_size = min(args.batch_size, test_instances - total)
        if loaded is None:
            problems = generate_problems(
                batch_size,
                args.problem_size,
                args.vehicle_types,
                device=device,
            )
        else:
            stop = total + batch_size
            problems = loaded.slice(total, stop)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        batch_start = time.perf_counter()
        costs, augmented_costs = evaluate_costs(
            model,
            problems,
            rollout_size=args.rollout_size,
            decode_type=args.decode_type,
            augment=args.augment,
            prune_dominated_types=args.prune_dominated_types,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_gpu_memory_mb = 0.0
        batch_seconds = time.perf_counter() - batch_start
        dominated_count = dominated_vehicle_mask(
            problems.vehicle_capacity,
            problems.vehicle_fixed_cost,
            problems.vehicle_variable_cost,
        ).sum(dim=1)
        costs_cpu = costs.cpu()
        augmented_cpu = augmented_costs.cpu()
        for offset in range(batch_size):
            rows.append(
                {
                    "dataset": dataset_name,
                    "instance_id": total + offset,
                    "method": args.method_name,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_epoch": checkpoint.get("epoch", ""),
                    "training_seed": checkpoint.get("run_config", {}).get("seed", ""),
                    "evaluation_seed": args.seed,
                    "problem_size": problems.problem_size,
                    "vehicle_types": problems.vehicle_type_num,
                    "dominated_type_count": int(dominated_count[offset].item()),
                    "rollout_size": args.rollout_size,
                    "decode_type": args.decode_type,
                    "augment": int(args.augment),
                    "prune_dominated_types": int(args.prune_dominated_types),
                    "cost": float(costs_cpu[offset].item()),
                    "augmented_cost": float(augmented_cpu[offset].item()),
                    "batch_size": batch_size,
                    "seconds_per_instance": batch_seconds / batch_size,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                }
            )
        cost_sum += float(costs_cpu.sum().item())
        augmented_sum += float(augmented_cpu.sum().item())
        total += batch_size
        print("evaluated={}/{}".format(total, test_instances), flush=True)
    evaluation_seconds = time.perf_counter() - evaluation_start
    print("mean_cost={:.6f}".format(cost_sum / total))
    print("mean_augmented_cost={:.6f}".format(augmented_sum / total))
    print("evaluation_seconds={:.3f}".format(evaluation_seconds))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary_path = args.output.with_suffix(".summary.json")
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "dataset": dataset_name,
                    "method": args.method_name,
                    "instances": total,
                    "mean_cost": cost_sum / total,
                    "mean_augmented_cost": augmented_sum / total,
                    "evaluation_seconds": evaluation_seconds,
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "gpu_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                },
                file,
                indent=2,
                sort_keys=True,
            )


if __name__ == "__main__":
    main()
