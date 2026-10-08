# Registered design of the version 2 experiments

Each cohort was registered before launch with its arms, seeds, call budgets, endpoints and tests; the registration files were fixed read-only on the server and are summarised here. Model: gpt-oss-20b (21B, 3.6B active) on a self-hosted vLLM endpoint, sampling temperature 0.8, operator output limit 2,048 tokens, host limit 8,192 tokens, population 8, five Weibull training instances (5,000 items, capacity 100), evaluation timeout 30 s, deterministic programs only. Sealed banks: `sealed_bank/sealed_c100.json` (capacity 100, seeds 92001-92005) and `sealed_c500.json` (capacity 500, seeds 92501-92505), generated from recorded seeds and hash-sealed (`seal.sha256`) before any cohort ran; never read during search or by any prompt; scored once per cohort after all runs finished.

| Cohort | Registered | Family | Arms | Calls per run | Seeds | Primary endpoint and test |
|---|---|---|---|---|---|---|
| hox2c | 2026-10-03 | steady-state | EoH 3200; Fermi-type T=0 + host; Fermi-type T=0.0003 + host | 3200 / 1600 + <= 1600 | 84001-84010 | sealed c100 at 800 total calls, T=0 + host vs EoH, two-sided exact Wilcoxon paired by seed (not met, p = 0.098); secondary at 200, 400, 1600 and endpoint |
| hox5 | 2026-10-05 | generational | Bose-type T = 0, 0.003, 0.03 + host | 177 x 18 = 3186 | 85001-85010 | training endpoint and sealed c100 vs T=0, exact Wilcoxon, Holm over the temperatures |
| hox2e, hox2f | 2026-10-06 | steady-state | Fermi-type T = 0.003, 0.03 + host | 1600 + <= 1600 | 84001-84010 | same seeds and tests as hox2c, vs T=0 |
| hox8 | 2026-10-06 | generational | Bose-type T = 0.0003 + host | 3186 | 85001-85010 | as hox5 |
| hox9 | 2026-10-06 | reach speed | A: EoH 800; B: Fermi-type T=0 + host; C: Fermi-type T=0.003 + host | 800; 400 + <= 400 | 86001-86010 (new) | sealed c100 at 400 total calls, B vs A (met: 9 wins of 10, p = 0.0039); secondary at 200 and at the endpoint <= 800 |
| hox10c, hox10b | 2026-10-06 | both | T = 0.1 + host | as the family | as the family | vs T=0 of the same family; degradation at high temperature as the hypothesis |
| hox11c1, hox11c2, hox11b1, hox11b2 | 2026-10-07 | both | T = 0.001, 0.01 + host | as the family | as the family | vs T=0 of the same family, Holm over the six temperatures per family and budget point; Spearman over the seven temperatures descriptive |

Budget-point selection: steady-state loop, the best training individual among registrations whose generation request started within the first b total calls (operator plus host); generational loop, the history-best individual up to generation floor(b / 18). Selection uses training values only; sealed values are reported for the selected individual. Runs are never rerun for outcome reasons; an infrastructure failure would be rerun once with the same seed and both runs reported. No run in these cohorts was rerun or excluded.
