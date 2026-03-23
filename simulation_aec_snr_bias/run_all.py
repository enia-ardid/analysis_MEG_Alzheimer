from __future__ import annotations

"""Entry point for the AEC-versus-AEC-orth SNR bias simulation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from simulation_aec_snr_bias.experiments import (
        DEFAULT_DELTA_TRIALS,
        DEFAULT_LIBRARY_SIZE,
        DEFAULT_RHO_LEVELS,
        DEFAULT_SNR_GRID,
        DEFAULT_SNR_LEVELS,
        EMPIRICAL_D,
        run_experiment_1_snr_sweep,
        run_experiment_2_trial_count_effect,
        run_experiment_3_snr_rho_interaction,
        run_targeted_trial_difference_scenario,
    )
    from simulation_aec_snr_bias.plotting import plot_experiment_1, plot_experiment_2, plot_experiment_3, plot_summary
else:
    from .experiments import (
        DEFAULT_DELTA_TRIALS,
        DEFAULT_LIBRARY_SIZE,
        DEFAULT_RHO_LEVELS,
        DEFAULT_SNR_GRID,
        DEFAULT_SNR_LEVELS,
        EMPIRICAL_D,
        run_experiment_1_snr_sweep,
        run_experiment_2_trial_count_effect,
        run_experiment_3_snr_rho_interaction,
        run_targeted_trial_difference_scenario,
    )
    from .plotting import plot_experiment_1, plot_experiment_2, plot_experiment_3, plot_summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the simulation suite."""
    parser = argparse.ArgumentParser(description="Run the AEC/AEC-orth SNR bias simulation.")
    parser.add_argument("--results-dir", default="simulation_aec_snr_bias/results", help="Directory used for tabular outputs.")
    parser.add_argument("--figures-dir", default="simulation_aec_snr_bias/figures", help="Directory used for PDF figures.")
    parser.add_argument("--random-seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib worker count for parameter sweeps.")
    parser.add_argument("--library-size", type=int, default=None, help="Number of precomputed trial metrics per (rho, SNR) condition.")
    parser.add_argument("--force", action="store_true", help="Recompute experiments even if CSV outputs already exist.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test version instead of the manuscript-scale simulation.")
    return parser.parse_args()


def _format_summary_table(rows: list[tuple[str, str, str]]) -> str:
    """Return a plain-text summary table."""
    width_question = max(len("Question"), *(len(row[0]) for row in rows))
    width_result = max(len("Result"), *(len(row[1]) for row in rows))
    width_interp = max(len("Interpretation"), *(len(row[2]) for row in rows))
    line = "+-" + "-" * width_question + "-+-" + "-" * width_result + "-+-" + "-" * width_interp + "-+"
    out = [line]
    out.append(f"| {'Question'.ljust(width_question)} | {'Result'.ljust(width_result)} | {'Interpretation'.ljust(width_interp)} |")
    out.append(line)
    for question, result, interpretation in rows:
        out.append(f"| {question.ljust(width_question)} | {result.ljust(width_result)} | {interpretation.ljust(width_interp)} |")
    out.append(line)
    return "\n".join(out)


def _required_snr(exp1_df: pd.DataFrame) -> float | None:
    """Return the minimum SNR whose mean d reaches the empirical effect."""
    crossing = exp1_df.loc[exp1_df["mean_cohen_d"] >= EMPIRICAL_D]
    if crossing.empty:
        return None
    return float(crossing.iloc[0]["snr"])


def _plausibility_judgment(required_snr: float | None) -> str:
    """Translate the required SNR into the requested plausibility statement."""
    if required_snr is None or required_snr > 8.0:
        return "NOT plausible: SNR artifact alone cannot explain the observed effect."
    if required_snr < 3.0:
        return "INCONCLUSIVE: cannot rule out SNR artifact."
    return "PLAUSIBLE: SNR artifact may contribute."


def _trial_diff_needed(exp2_df: pd.DataFrame) -> str:
    """Return the trial-count difference required to reach the empirical effect."""
    crossing = exp2_df.loc[exp2_df["mean_cohen_d"] >= EMPIRICAL_D]
    if crossing.empty:
        return f">{int(exp2_df['delta_trials'].max())} trials"
    return f"{int(crossing.iloc[0]['delta_trials'])} trials"


def _load_or_build_frame(csv_path: Path, force: bool, builder) -> pd.DataFrame:
    """Load an existing CSV result or build it and save it."""
    if csv_path.exists() and not force:
        return pd.read_csv(csv_path)
    frame = builder()
    frame.to_csv(csv_path, index=False)
    return frame


def main() -> None:
    """Run all experiments, save the outputs, and print the summary table."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        exp1_iter, exp2_iter, exp3_iter, summary_iter = 2, 2, 1, 2
        library_size = 256 if args.library_size is None else args.library_size
    else:
        exp1_iter, exp2_iter, exp3_iter, summary_iter = 2000, 2000, 1000, 1000
        library_size = DEFAULT_LIBRARY_SIZE if args.library_size is None else args.library_size

    exp1_path = results_dir / "experiment_1_snr_sweep.csv"
    exp2_path = results_dir / "experiment_2_trial_count_effect.csv"
    exp3_path = results_dir / "experiment_3_snr_rho_interaction.csv"
    targeted_path = results_dir / "summary_targeted_delta_trials.csv"

    exp1 = _load_or_build_frame(
        exp1_path,
        args.force,
        lambda: run_experiment_1_snr_sweep(
            n_iter=exp1_iter,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
            library_size=library_size,
        ),
    )
    exp2 = _load_or_build_frame(
        exp2_path,
        args.force,
        lambda: run_experiment_2_trial_count_effect(
            n_iter=exp2_iter,
            random_seed=args.random_seed + 1,
            n_jobs=args.n_jobs,
            library_size=library_size,
        ),
    )
    exp3 = _load_or_build_frame(
        exp3_path,
        args.force,
        lambda: run_experiment_3_snr_rho_interaction(
            n_iter=exp3_iter,
            random_seed=args.random_seed + 2,
            n_jobs=args.n_jobs,
            library_size=library_size,
        ),
    )

    if targeted_path.exists() and not args.force:
        targeted = pd.read_csv(targeted_path).iloc[0].to_dict()
    else:
        targeted = run_targeted_trial_difference_scenario(
            snr=5.0,
            delta_trials=3,
            n_iter=summary_iter,
            random_seed=args.random_seed + 3,
            library_size=library_size,
        )
        pd.DataFrame([targeted]).to_csv(targeted_path, index=False)

    np.savez(
        results_dir / "simulation_results.npz",
        experiment_1=exp1.to_records(index=False),
        experiment_2=exp2.to_records(index=False),
        experiment_3=exp3.to_records(index=False),
        targeted_summary=np.array([tuple(targeted.values())], dtype=object),
        snr_grid=DEFAULT_SNR_GRID,
        delta_trials=DEFAULT_DELTA_TRIALS,
        rho_levels=DEFAULT_RHO_LEVELS,
        snr_levels=DEFAULT_SNR_LEVELS,
    )

    plot_experiment_1(exp1, figures_dir / "figure_1_snr_sweep.pdf")
    plot_experiment_2(exp2, figures_dir / "figure_2_trial_count_effect.pdf")
    plot_experiment_3(exp3, figures_dir / "figure_3_snr_rho_heatmap.pdf")
    plot_summary(exp1, figures_dir / "figure_4_summary.pdf")

    required_snr = _required_snr(exp1)
    judgment = _plausibility_judgment(required_snr)
    trial_diff_needed = _trial_diff_needed(exp2)
    accounted_pct = 100.0 * float(targeted["mean_cohen_d"]) / EMPIRICAL_D

    rows = [
        (
            "Min SNR to reproduce empirical d=0.575",
            "not reached" if required_snr is None else f"{required_snr:.2f}",
            judgment,
        ),
        (
            "d from 3-trial diff at SNR=5",
            f"{float(targeted['mean_cohen_d']):.2f}",
            f"accounts for {accounted_pct:.0f}% of observed effect",
        ),
        (
            "Trial diff needed to reach d=0.575",
            trial_diff_needed,
            "versus observed 3 trials",
        ),
    ]

    print("\nSIMULATION SUMMARY")
    print(_format_summary_table(rows))


if __name__ == "__main__":
    main()
