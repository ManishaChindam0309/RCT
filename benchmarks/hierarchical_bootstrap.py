
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "rct_adaptive_prefix_gate_confirmation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "rct_hierarchical_bootstrap"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42

REQUIRED_COLUMNS = {
    "batch_size",
    "prefix_length",
    "prompt_length",
    "new_tokens",
    "repetition",
    "kv_ms",
    "auto_ms",
    "paired_speedup",
}


# ---------------------------------------------------------
# Load all benchmark sessions
# ---------------------------------------------------------

def load_sessions():
    session_data = {}

    session_dirs = sorted(
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    )

    if not session_dirs:
        raise RuntimeError(
            f"No benchmark sessions found in {RESULTS_DIR}"
        )

    for session_dir in session_dirs:
        csv_path = session_dir / "raw_measurements.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {sorted(missing)}"
            )

        numeric_columns = [
            "batch_size",
            "prefix_length",
            "prompt_length",
            "new_tokens",
            "repetition",
            "kv_ms",
            "auto_ms",
            "paired_speedup",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column], errors="raise"
            )

        if df[numeric_columns].isna().any().any():
            raise ValueError(
                f"Missing numeric values in {csv_path}"
            )

        if (df["kv_ms"] <= 0).any() or (df["auto_ms"] <= 0).any():
            raise ValueError(
                f"Non-positive latency found in {csv_path}"
            )

        # Verify the stored paired speedup.
        calculated = df["kv_ms"] / df["auto_ms"]

        if not np.allclose(
            calculated,
            df["paired_speedup"],
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError(
                f"paired_speedup does not match kv_ms / auto_ms "
                f"in {csv_path}"
            )

        df["session"] = session_dir.name

        session_data[session_dir.name] = df

        print(
            f"Loaded {session_dir.name}: "
            f"{len(df)} measurements"
        )

    print(f"\nTotal sessions loaded: {len(session_data)}")

    return session_data


# ---------------------------------------------------------
# Hierarchical bootstrap
# ---------------------------------------------------------

def hierarchical_bootstrap(groups, iterations, rng):
    """
    First resample benchmark sessions with replacement.
    Then resample repetitions within each selected session.

    Each selected session contributes the same number of
    observations as its original workload group.
    """

    session_names = list(groups.keys())

    if len(session_names) < 2:
        raise ValueError(
            "At least two independent sessions are required."
        )

    original_values = np.concatenate(
        [groups[name] for name in session_names]
    )

    point_estimate = float(np.median(original_values))

    bootstrap_medians = np.empty(iterations)

    for i in range(iterations):
        selected_sessions = rng.choice(
            session_names,
            size=len(session_names),
            replace=True,
        )

        sampled_values = []

        for session_name in selected_sessions:
            values = groups[session_name]

            sampled = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            sampled_values.append(sampled)

        combined = np.concatenate(sampled_values)

        bootstrap_medians[i] = np.median(combined)

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return {
        "median_speedup": point_estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_iterations": iterations,
    }


# ---------------------------------------------------------
# Analyze workloads
# ---------------------------------------------------------

def analyze(session_data):
    workload_columns = [
        "batch_size",
        "prefix_length",
        "prompt_length",
        "new_tokens",
    ]

    all_data = pd.concat(
        session_data.values(),
        ignore_index=True,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    workload_keys = (
        all_data[workload_columns]
        .drop_duplicates()
        .sort_values(workload_columns)
    )

    for _, workload in workload_keys.iterrows():

        mask = np.ones(len(all_data), dtype=bool)

        for column in workload_columns:
            mask &= (
                all_data[column].to_numpy()
                == workload[column]
            )

        workload_df = all_data.loc[mask]

        groups = {}

        for session_name, session_df in workload_df.groupby("session"):
            values = session_df["paired_speedup"].to_numpy(
                dtype=float
            )

            if len(values) == 0:
                continue

            groups[session_name] = values

        if len(groups) != len(session_data):
            raise ValueError(
                "A workload is missing from one or more sessions: "
                f"{workload.to_dict()}"
            )

        bootstrap = hierarchical_bootstrap(
            groups,
            BOOTSTRAP_ITERATIONS,
            rng,
        )

        run_medians = {
            name: float(np.median(values))
            for name, values in groups.items()
        }

        run_medians_favor_auto = sum(
            median > 1.0
            for median in run_medians.values()
        )

        result = {
            "batch_size": int(workload["batch_size"]),
            "prefix_length": int(workload["prefix_length"]),
            "prompt_length": int(workload["prompt_length"]),
            "new_tokens": int(workload["new_tokens"]),
            "sessions": len(groups),
            "measurements": int(len(workload_df)),
            "median_speedup": bootstrap["median_speedup"],
            "ci95_lower": bootstrap["ci95_lower"],
            "ci95_upper": bootstrap["ci95_upper"],
            "run_medians_favor_auto": run_medians_favor_auto,
            "run_medians_favor_kv": (
                len(groups) - run_medians_favor_auto
            ),
            "conclusion": (
                "AUTO FASTER"
                if bootstrap["ci95_lower"] > 1.0
                else (
                    "KV FASTER"
                    if bootstrap["ci95_upper"] < 1.0
                    else "INCONCLUSIVE"
                )
            ),
        }

        results.append(result)

        print(
            f"Batch {result['batch_size']}, "
            f"Prefix {result['prefix_length']}: "
            f"{result['median_speedup']:.4f}x "
            f"[{result['ci95_lower']:.4f}, "
            f"{result['ci95_upper']:.4f}] "
            f"{result['conclusion']}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("RCT HIERARCHICAL BOOTSTRAP ANALYSIS")
    print("=" * 65)

    print(f"Results directory: {RESULTS_DIR}")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    session_data = load_sessions()

    print("\nAnalyzing workloads...\n")

    results = analyze(session_data)

    output_path = OUTPUT_DIR / "hierarchical_bootstrap_results.csv"

    results.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        results[
            [
                "batch_size",
                "prefix_length",
                "median_speedup",
                "ci95_lower",
                "ci95_upper",
                "run_medians_favor_auto",
                "conclusion",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved results to:\n{output_path}")


if __name__ == "__main__":
    main()
