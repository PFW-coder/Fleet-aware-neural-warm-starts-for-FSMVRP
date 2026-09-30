import unittest
from pathlib import Path

import torch

from fsmvrp.env import FSMVRPEnv
from fsmvrp.model import scaled_dot_product_attention
from fsmvrp.problem import ProblemBatch, generate_fsmfd_v2_problems
from fsmvrp.runner import evaluate_costs, load_model, seed_everything


def tiny_problem() -> ProblemBatch:
    return ProblemBatch(
        depot_xy=torch.tensor([[[0.0, 0.0]]]),
        node_xy=torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]),
        node_demand=torch.tensor([[0.6, 0.6]]),
        vehicle_capacity=torch.tensor([[1.0]]),
        vehicle_fixed_cost=torch.tensor([[5.0]]),
        vehicle_variable_cost=torch.tensor([[2.0]]),
    )


class CoreTests(unittest.TestCase):
    def test_fixed_and_variable_cost(self):
        env = FSMVRPEnv(rollout_size=1)
        env.reset(tiny_problem())
        env.step(torch.tensor([[1]]))
        env.step(torch.tensor([[0]]))
        _, reward, done = env.step(torch.tensor([[2]]))
        self.assertTrue(done)
        self.assertAlmostEqual(float((-reward).item()), 22.0)

    def test_attention_normalizes_per_query(self):
        query = torch.zeros(1, 1, 1, 2, 4)
        key = torch.zeros(1, 1, 1, 3, 4)
        value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 1, 3, 4)
        _, weight = scaled_dot_product_attention(
            query, key, value, return_weights=True
        )
        self.assertTrue(torch.allclose(weight.sum(dim=-1), torch.ones(1, 1, 1, 2)))

    def test_released_checkpoint_runs(self):
        seed_everything(7, deterministic=True)
        checkpoint = Path(__file__).parents[1] / "checkpoints" / "fsmvrp_fleet_aware.pt"
        model, metadata = load_model(str(checkpoint), torch.device("cpu"))
        problems = generate_fsmfd_v2_problems(1, 8, 3)
        cost, _ = evaluate_costs(
            model,
            problems,
            rollout_size=1,
            decode_type="greedy",
            prune_dominated_types=True,
        )
        self.assertEqual(metadata["format_version"], 1)
        self.assertTrue(torch.isfinite(cost).all())


if __name__ == "__main__":
    unittest.main()
