# Reference implementations (require the full run records)

These scripts document how the released aggregates were produced. They read the
complete run records (candidate lineage `result_bp_online_seed101.json`,
`registration.jsonl`) and, for the population evaluations, re-execute candidate
programs in the sandbox. Those raw records are not part of this release. Their outputs are released in
`results/paper_v1/` and in each run directory (`pop_endpoint.json`,
`g1_history_best.json`).

| Script | Produces |
|---|---|
| `pop_c500*.py` | `pop_endpoint.json` per run (final population scored on the capacity-500 instances), one script per run family |
| `g1_history_best*.py` | `g1_history_best.json` per generational run (history-best candidate re-scored) |
| `make_fig2_table6.py` | best-so-far curves (`fig2_best_so_far_*.json`, `per_run_best_so_far_*.json`) and operator outcome tables (`table6_*.json`, `per_run_operator_outcomes_*.json`) |
