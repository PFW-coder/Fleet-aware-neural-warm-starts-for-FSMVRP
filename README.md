# Fleet-aware neural warm starts for FSMVRP

This repository contains the public implementation of a fleet-aware neural construction policy for the fleet size and mix vehicle routing problem (FSMVRP). The policy jointly selects a reusable vehicle type and the next customer, accounts for fixed deployment and type-dependent travel costs, and can provide a warm start to PyVRP under a shared wall-time budget.

The release is intentionally compact. It contains the core model, training and evaluation entry points, the hybrid solver, one inference checkpoint, focused tests, and selected aggregate results. Internal experiment orchestration, raw route files, logs, and development artifacts are not included.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional PyVRP hybrid solver with:

```bash
python -m pip install -e '.[hybrid]'
```

## Quick evaluation

Generate a deterministic test set:

```bash
python generate_data.py \
  --instances 16 \
  --problem-size 100 \
  --vehicle-types 9 \
  --distribution fsmfd_v2 \
  --seed 1234 \
  --output data/smoke.pt
```

Evaluate the released model:

```bash
python evaluate.py \
  --checkpoint checkpoints/fsmvrp_fleet_aware.pt \
  --data data/smoke.pt \
  --batch-size 1 \
  --rollout-size 20 \
  --decode-type sampling \
  --augment \
  --prune-dominated-types \
  --device cuda \
  --output outputs/neural.csv
```

Use `--device cpu` when CUDA is unavailable. For large instances, reduce the rollout size and use the hybrid solver's `--augmentation-chunk-size` option to control GPU memory.

## Hybrid solver

```bash
python -m baselines.run_hybrid \
  --checkpoint checkpoints/fsmvrp_fleet_aware.pt \
  --data data/smoke.pt \
  --rollout-size 20 \
  --decode-type sampling \
  --augment \
  --augmentation-chunk-size 1 \
  --prune-dominated-types \
  --max-total-runtime 12 \
  --seed 777 \
  --device cuda \
  --output outputs/hybrid.csv
```

The total-runtime mode charges neural construction, PyVRP setup, and PyVRP search to the same per-instance wall-time budget. Returned routes are revalidated against the original continuous FSMVRP data.

## Training

Single-GPU example:

```bash
python train.py \
  --problem-sizes 100,200,300 \
  --vehicle-type-options 3,5,9 \
  --training-distribution fsmfd_mix \
  --rollout-size 10 \
  --epochs 120 \
  --episodes-per-epoch 120 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lr-milestones 80,100 \
  --prune-dominated-types \
  --device cuda \
  --output runs/fsmvrp
```

Distributed training uses `torchrun`; `--batch-size` and `--episodes-per-epoch` are global values.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --problem-sizes 100,200,300 \
  --vehicle-type-options 3,5,9 \
  --training-distribution fsmfd_mix \
  --rollout-size 10 \
  --epochs 120 \
  --episodes-per-epoch 120 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lr-milestones 80,100 \
  --prune-dominated-types \
  --device cuda \
  --output runs/fsmvrp_ddp
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Results

Selected aggregate tables are provided in `results/`. Negative `hybrid_minus_cold_pct` values indicate lower cost for the neural-warm-start hybrid than cold-start PyVRP. These tables are intended for result inspection; they are not a substitute for the full experimental archive.

## License

This code is released under the MIT License.
