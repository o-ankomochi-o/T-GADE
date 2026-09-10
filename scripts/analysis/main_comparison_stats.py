"""Main-comparison statistics from the released run table (read-only).

Reads results/paper_v1/run_outcomes_v1.csv and prints, for the generational
Bose-type T=0.003 configuration against the EoH reference (20 runs each):
medians, Mann-Whitney U (two-sided and one-sided, lower excess is better),
Cliff's delta, and the <=1.0% attainment counts with a one-sided Fisher test.
Rank statistics use outcomes rounded to five decimal places (the recorded
objective precision); medians use the unrounded values.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results/paper_v1/run_outcomes_v1.csv"
MAIN = "G_boson_T0.003"
REF = "EoH"
BASELINES = ("B1_parent_free", "B2_quality_blind")


def load() -> dict[str, list[tuple[int, float]]]:
    arms: dict[str, list[tuple[int, float]]] = {}
    with CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            arms.setdefault(row["arm"], []).append((int(row["client_seed"]), float(row["train_excess_pct"])))
    return arms


def cliff_delta(a: list[float], b: list[float]) -> float:
    """Positive when values in `a` tend to be lower (better) than in `b`."""
    lower = sum(x < y for x in a for y in b)
    higher = sum(x > y for x in a for y in b)
    return (lower - higher) / (len(a) * len(b))


def main() -> None:
    arms = load()
    g = [v for _, v in arms[MAIN]]
    e = [v for _, v in arms[REF]]
    assert len(g) == len(e) == 20, (len(g), len(e))
    gr = [round(v / 100, 5) for v in g]
    er = [round(v / 100, 5) for v in e]
    two = mannwhitneyu(gr, er, alternative="two-sided", method="asymptotic")
    one = mannwhitneyu(gr, er, alternative="less", method="asymptotic")
    att_g = sum(v <= 1.0 for v in g)
    att_e = sum(v <= 1.0 for v in e)
    _, fisher_p = fisher_exact([[att_g, 20 - att_g], [att_e, 20 - att_e]], alternative="greater")
    print(f"{'quantity':<42}{MAIN:>18}{REF:>12}")
    print(f"{'median training excess (%)':<42}{statistics.median(g):>18.4f}{statistics.median(e):>12.4f}")
    print(f"{'runs with excess <= 1.0%':<42}{att_g:>18d}{att_e:>12d}")
    print(f"Mann-Whitney U = {two.statistic:.1f}, two-sided p = {two.pvalue:.4f}, one-sided p = {one.pvalue:.4f}")
    print(f"Cliff's delta (lower is better) = {cliff_delta(gr, er):.3f}")
    print(f"Fisher one-sided p for <= 1.0% attainment = {fisher_p:.4f}")
    ties = sum(x == y for x in gr for y in er)
    print(f"tied pairs at five-decimal precision = {ties}")
    seeds_b = {s for s, _ in arms[BASELINES[0]]}
    e_sub = [v for s, v in arms[REF] if s in seeds_b]
    print(f"{'baseline medians (10 seeds)':<42}", end="")
    for name in BASELINES:
        print(f"{name} = {statistics.median(v for _, v in arms[name]):.3f}  ", end="")
    print(f"{REF} (same seeds) = {statistics.median(e_sub):.3f}")


if __name__ == "__main__":
    main()
