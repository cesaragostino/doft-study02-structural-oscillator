"""Generate figures for integer participation vs null models and noise coupling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_data(participation_csv: Path, materials_csv: Path, noise_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    part = pd.read_csv(participation_csv)
    materials = pd.read_csv(materials_csv)
    noise = pd.read_csv(noise_csv) if noise_csv else pd.DataFrame()
    if "material" in noise.columns:
        noise = noise.rename(columns={"material": "name"})
    merged = materials.merge(noise, on="name", how="left", suffixes=("", "_noise"))
    merged["predicted_noise"] = pd.to_numeric(merged.get("predicted_noise"), errors="coerce")
    merged["ThetaD_K"] = pd.to_numeric(merged.get("ThetaD_K"), errors="coerce")
    merged["Tc_K"] = pd.to_numeric(merged.get("Tc_K"), errors="coerce")
    merged["Tc_ideal"] = merged["Tc_K"] * (1.0 + merged["predicted_noise"].fillna(0.0))
    merged["category"] = merged.get("category", merged.get("category_noise", "Unknown")).fillna("Unknown")
    part["category"] = part.get("category", "Unknown").fillna("Unknown")
    return part, merged


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    ranks = []
    for x in a:
        ranks.append(np.sum(x > b) - np.sum(x < b))
    return float(np.sum(ranks) / (len(a) * len(b)))


def generate_shuffle_once(theta: np.ndarray, tc_ideal: np.ndarray, f_base_arr: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    perm_theta = rng.permutation(theta)
    fm_perm = perm_theta / tc_ideal
    n_perm = fm_perm / f_base_arr
    delta_perm = np.abs(n_perm - np.round(n_perm))
    return n_perm, delta_perm


def plot_figures(
    part: pd.DataFrame,
    merged: pd.DataFrame,
    output_dir: Path,
    n_null: int = 200,
    max_n_plot: float = 40.0,
    delta_cap: float = 1.0,
    seed: int = 123,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Align participation with materials for f_base_used and Tc_ideal/ThetaD
    df = part.merge(merged[["name", "ThetaD_K", "Tc_ideal", "category"]], on="name", how="left", suffixes=("", "_mat"))
    df["f_base_used"] = pd.to_numeric(df.get("f_base_used"), errors="coerce")
    df["N_value"] = pd.to_numeric(df.get("N_value"), errors="coerce")
    df["delta_value"] = pd.to_numeric(df.get("delta_value"), errors="coerce")
    df["z_noise"] = pd.to_numeric(df.get("z_noise"), errors="coerce")
    df = df[df["N_value"] >= 1.0]

    theta = df["ThetaD_K"].to_numpy(dtype=float)
    tc_ideal = df["Tc_ideal"].to_numpy(dtype=float)
    f_base_arr = df["f_base_used"].to_numpy(dtype=float)

    # Null sample (representative shuffle)
    n_null_sample, delta_null_sample = generate_shuffle_once(theta, tc_ideal, f_base_arr, rng)
    delta_real = df["delta_value"].to_numpy(dtype=float)
    ks_stat, ks_p = stats.ks_2samp(delta_real, delta_null_sample)
    mw_stat, mw_p = stats.mannwhitneyu(delta_real, delta_null_sample, alternative="less")
    cliffs_real_null = _cliffs_delta(-delta_real, -delta_null_sample)  # negative means real < null

    # Figure 1A: N histogram real vs shuffle
    plt.figure(figsize=(8, 5))
    bins_n = np.linspace(0, max_n_plot, 60)
    plt.hist(df["N_value"], bins=bins_n, alpha=0.6, label="Real", density=True, color="#2b8cbe")
    plt.hist(n_null_sample, bins=bins_n, alpha=0.5, label="Null (shuffle ΘD)", density=True, color="#f03b20")
    for vline in [1, 2, 3, 24, 25]:
        if vline <= max_n_plot:
            plt.axvline(vline, color="k", linestyle=":", linewidth=0.8)
    plt.xlabel("N_corrected = Fm*/f_base")
    plt.ylabel("Density")
    plt.title("Integer participation vs null (N)")
    plt.legend()
    fig1a = output_dir / "fig01a_hist_N_real_vs_shuffle.png"
    plt.tight_layout()
    plt.savefig(fig1a, dpi=200)
    plt.close()

    # Figure 1B: |delta| histogram real vs shuffle (log y)
    plt.figure(figsize=(8, 5))
    bins_d = np.linspace(0, delta_cap, 60)
    counts_real, edges_real = np.histogram(df["delta_value"], bins=bins_d, density=True)
    plt.hist(df["delta_value"], bins=bins_d, alpha=0.7, label="Real", density=True, color="#2b8cbe")
    plt.hist(delta_null_sample, bins=bins_d, alpha=0.5, label="Null (shuffle ΘD)", density=True, color="#f03b20")
    plt.yscale("log")
    plt.xlabel("|delta| = |N_corrected - round(N_corrected)|")
    plt.ylabel("Density (log scale)")
    plt.title("Integer participation vs null (|delta|)")
    plt.legend()
    if counts_real.size:
        peak_idx = int(np.argmax(counts_real))
        peak_x = (edges_real[peak_idx] + edges_real[peak_idx + 1]) / 2
        peak_y = max(counts_real[peak_idx], 1e-9)
        plt.annotate(
            "Significant Quantization (p < 0.001)",
            xy=(peak_x, peak_y),
            xytext=(peak_x + 0.1, peak_y * 5),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.0),
            fontsize=9,
            color="k",
        )
    plt.text(0.02, plt.ylim()[1] * 0.4, f"KS p={ks_p:.3e}\nMW p={mw_p:.3e}\nCliff's Δ={cliffs_real_null:.2f}", fontsize=9)
    fig1b = output_dir / "fig01b_hist_delta_real_vs_shuffle.png"
    plt.tight_layout()
    plt.savefig(fig1b, dpi=200)
    plt.close()

    # Figure 2: delta by family
    df_fam = df.copy()
    df_fam["delta_clipped"] = df_fam["delta_value"].clip(upper=delta_cap)
    family_labels = {
        "SC_TypeI": "SC_Type-I",
        "SC_TypeII": "SC_Type-II",
        "SC_IronBased": "SC_IronBased",
        "SC_HighPressure": "SC_HighPressure",
        "SC_Binary": "SC_Binary",
        "SC_HeavyFermion": "SC_HeavyFermion",
        "SC_Molecular": "SC_Molecular",
        "SC_Oxide": "SC_Oxide",
        "Superfluid": "Superfluid",
    }
    df_fam["family_label"] = df_fam["category"].apply(lambda x: family_labels.get(x, str(x)))
    order = df_fam.groupby("family_label")["delta_clipped"].median().sort_values().index.tolist()
    data_ordered = [df_fam.loc[df_fam["family_label"] == fam, "delta_clipped"].dropna().to_numpy() for fam in order]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data_ordered, tick_labels=order, whis=[5, 95], showfliers=False, patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("|delta| (clipped at 1.0)")
    plt.title("Per-family integer locking strength (ordered by median)")
    fig2 = output_dir / "fig02_delta_by_family.png"
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    # Figure 3A: scatter delta vs z_noise
    plt.figure(figsize=(8, 5))
    categories = df["category"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    for col, cat in zip(colors, categories):
        mask = df["category"] == cat
        plt.scatter(df.loc[mask, "z_noise"], df.loc[mask, "delta_value"], alpha=0.45, s=18, label=cat, color=col)
    rho, pval = stats.spearmanr(df["delta_value"], df["z_noise"], nan_policy="omit")
    plt.xlabel("Z-score noise")
    plt.ylabel("|delta|")
    plt.title("Noise vs integer locking")
    plt.legend(fontsize="x-small", ncol=2)
    plt.text(0.05, plt.ylim()[1] * 0.8, f"Spearman ρ={rho:.3f}, p={pval:.3e}", fontsize=9)
    fig3a = output_dir / "fig03a_delta_vs_noise_scatter.png"
    plt.tight_layout()
    plt.savefig(fig3a, dpi=200)
    plt.close()

    # Figure 3B: noise for almost-integer vs rest
    cutoff = df["delta_value"].quantile(0.2)
    near = df[df["delta_value"] <= cutoff]["z_noise"].to_numpy(dtype=float)
    rest = df[df["delta_value"] > cutoff]["z_noise"].to_numpy(dtype=float)
    cliffs = _cliffs_delta(near, rest)
    plt.figure(figsize=(6, 5))
    plt.boxplot([near, rest], tick_labels=[f"Almost integer (<=p20, Δ={cliffs:.2f})", "Rest"], showfliers=False)
    plt.ylabel("Z-score noise")
    plt.title("Noise vs almost-integer group")
    fig3b = output_dir / "fig03b_noise_almost_integer_vs_rest.png"
    plt.tight_layout()
    plt.savefig(fig3b, dpi=200)
    plt.close()

    # Figure 4A: f_base by family
    if "f_base_used" in df.columns:
        fam_label_map: Dict[str, str] = {
            "SC_TypeI": "SC_Type-I",
            "SC_TypeII": "SC_Type-II",
            "SC_IronBased": "SC_IronBased",
            "SC_HighPressure": "SC_HighPressure",
            "SC_Binary": "SC_Binary",
            "SC_HeavyFermion": "SC_HeavyFermion",
            "SC_Molecular": "SC_Molecular",
            "SC_Oxide": "SC_Oxide",
            "Superfluid": "Superfluid",
        }
        fam_fbase = df.groupby("category")["f_base_used"].median()
        fam_fbase.index = fam_fbase.index.map(lambda x: fam_label_map.get(x, str(x)))
        fam_fbase = fam_fbase.sort_values()
        plt.figure(figsize=(10, 5))
        plt.bar(fam_fbase.index, fam_fbase.values, color="#2b8cbe")
        global_line = df["f_base_used"].median()
        plt.axhline(global_line, color="k", linestyle="--", label=f"Global median={global_line:.3f}")
        plt.yscale("log")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("f_base_used (median per family)")
        plt.title("Base frequency by family (ordered by median)")
        plt.legend()
        fig4a = output_dir / "fig04a_fbase_by_family.png"
        plt.tight_layout()
        plt.savefig(fig4a, dpi=200)
        plt.close()

    print(f"[INFO] Figures written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot integer participation figures (v4.0).")
    parser.add_argument("--participation-csv", type=Path, required=True)
    parser.add_argument("--materials-csv", type=Path, required=True)
    parser.add_argument("--noise-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/participation_figures"))
    parser.add_argument("--n-null", type=int, default=200, help="Null permutations for representative shuffle")
    parser.add_argument("--max-n-plot", type=float, default=40.0)
    parser.add_argument("--delta-cap", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part, merged = load_data(args.participation_csv, args.materials_csv, args.noise_csv)
    plot_figures(
        part=part,
        merged=merged,
        output_dir=args.output_dir,
        n_null=args.n_null,
        max_n_plot=args.max_n_plot,
        delta_cap=args.delta_cap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
