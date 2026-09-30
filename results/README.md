# Selected aggregate results

- `fleet_complexity.csv`: three-seed paired results grouped by the number of available vehicle types.
- `k30_scaling.csv`: three-seed paired results for 30 vehicle types grouped by problem size.
- `sweep_control.csv`: comparison with a deterministic type-aware sweep warm start.
- `time_budget.csv`: paired hybrid and cold-start PyVRP results across wall-time budgets.
- `synthetic_equal_time.csv`: equal-time synthetic validation summary.
- `public_fsmfd_equal_time.csv`: equal-time public FSMFD benchmark summary.
- `golden_matched_time.csv`: matched-time Golden-instance summary.

Percentage differences use the cold-start solution as the denominator. A negative hybrid-minus-cold value means the neural-warm-start hybrid has lower cost.
