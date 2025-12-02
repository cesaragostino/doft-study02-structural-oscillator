"""Generate figures for integer participation vs null models and noise coupling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

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


def build_dataset_summary(materials: pd.DataFrame, participation: Optional[pd.DataFrame], output_dir: Path) -> None:
    def _count_contains(series: pd.Series, keyword: str) -> int:
        return series.fillna("").str.contains(keyword, case=False).sum()

    rows = []
    for family, g in materials.groupby(materials["category"].fillna("Unknown")):
        names = g["name"].astype(str)
        subnets = g["sub_network"].astype(str)
        row = {
            "family": family,
            "N_materials": names.nunique(),
            "N_rows": len(g),
            "N_single": int((subnets == "single").sum()),
            "N_sigma": int(_count_contains(subnets, "sigma")),
            "N_pi": int(_count_contains(subnets, "pi")),
            "N_pressure_modes": int(_count_contains(subnets, "bar")),
            "Tc_median": float(pd.to_numeric(g["Tc_K"], errors="coerce").median()),
            "ThetaD_median": float(pd.to_numeric(g["ThetaD_K"], errors="coerce").median()),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(by="family")

    if participation is not None and not participation.empty:
        part_family = participation.groupby("category").agg(
            N_mean=("N_value", "mean"),
            delta_median=("delta_value", "median"),
        )
        summary_df = summary_df.merge(part_family, left_on="family", right_index=True, how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "table_dataset_summary.csv", index=False)


def plot_family_hist_delta(df: pd.DataFrame, family: str, output_dir: Path, rng: np.random.Generator) -> None:
    fam_df = df[df["category"] == family]
    if fam_df.empty:
        return
    theta = fam_df["ThetaD_K"].to_numpy(dtype=float)
    tc_ideal = fam_df["Tc_ideal"].to_numpy(dtype=float)
    f_base_arr = fam_df["f_base_used"].to_numpy(dtype=float)
    _, delta_null = generate_shuffle_once(theta, tc_ideal, f_base_arr, rng)
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, max(fam_df["delta_value"].max(), delta_null.max()) if len(delta_null) else 0.5, 50)
    plt.hist(fam_df["delta_value"], bins=bins, density=True, alpha=0.7, label="Real", color="#2b8cbe")
    plt.hist(delta_null, bins=bins, density=True, alpha=0.5, label="Null (shuffle ΘD)", color="#f03b20")
    plt.xlabel("|N - round(N)|")
    plt.ylabel("Probability density")
    plt.title(f"Family {family}: integer residuals real vs null")
    plt.legend()
    fig_path = output_dir / f"fig_integer_hist_{family}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_delta_vs_znoise(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    categories = df["category"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    for col, cat in zip(colors, categories):
        mask = df["category"] == cat
        plt.scatter(df.loc[mask, "delta_value"], df.loc[mask, "z_noise"], alpha=0.5, s=18, label=cat, color=col)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    rho, pval = stats.spearmanr(df["delta_value"], df["z_noise"], nan_policy="omit")
    plt.xlabel("Integer residual δ")
    plt.ylabel("Normalised structural noise Zξ")
    plt.title("Integer residual vs noise (all families)")
    plt.legend(fontsize="x-small", ncol=2)
    plt.text(0.02, plt.ylim()[1] * 0.8, f"Spearman ρ={rho:.3f}, p={pval:.3e}", fontsize=9)
    out_path = output_dir / "fig_delta_vs_zxi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_curves(df: pd.DataFrame, families: List[str], output_dir: Path, lambda_penalty: float, bounds: Tuple[float, float]) -> None:
    if not families:
        return
    plt.figure(figsize=(8, 5))
    f_grid = np.geomspace(bounds[0], bounds[1], 300)
    for fam in families:
        fam_df = df[df["category"] == fam]
        if fam_df.empty:
            continue
        fm_corr = fam_df["Fm_corr"].to_numpy(dtype=float)
        fm_corr = fm_corr[np.isfinite(fm_corr) & (fm_corr > 0)]
        if fm_corr.size == 0:
            continue
        losses = []
        for f in f_grid:
            n_vals = fm_corr / f
            delta = np.abs(n_vals - np.round(n_vals))
            med_delta = np.median(delta)
            mean_n = np.mean(n_vals)
            losses.append(med_delta + lambda_penalty * mean_n)
        plt.plot(f_grid, losses, label=fam)
        # mark minimum
        idx = int(np.argmin(losses))
        plt.axvline(f_grid[idx], color=plt.gca().lines[-1].get_color(), linestyle="--", alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Base frequency f")
    plt.ylabel("Penalised loss L̃(f)")
    plt.title("Loss curves by family")
    plt.legend()
    out_path = output_dir / "fig_Lk_vs_f.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_figures(
    part: pd.DataFrame,
    merged: pd.DataFrame,
    output_dir: Path,
    n_null: int = 200,
    max_n_plot: float = 40.0,
    delta_cap: float = 1.0,
    family_hist: str = "SC_Binary",
    loss_families: Optional[List[str]] = None,
    lambda_penalty: float = 0.001,
    bounds: Tuple[float, float] = (0.5, 5.0),
    seed: int = 123,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Align participation with materials for f_base_used and Tc_ideal/ThetaD
    df = part.merge(merged[["name", "ThetaD_K", "Tc_K", "Tc_ideal", "category"]], on="name", how="left", suffixes=("", "_mat"))
    df["f_base_used"] = pd.to_numeric(df.get("f_base_used"), errors="coerce")
    df["N_value"] = pd.to_numeric(df.get("N_value"), errors="coerce")
    df["delta_value"] = pd.to_numeric(df.get("delta_value"), errors="coerce")
    df["z_noise"] = pd.to_numeric(df.get("z_noise"), errors="coerce")
    df = df[df["N_value"] >= 1.0]

    # Dataset summary table
    build_dataset_summary(merged, part, output_dir)

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
    plt.ylabel("Probability density")
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
    near = df[df["delta_value"] <= cutoff]["z_noise"].dropna().to_numpy(dtype=float)
    rest = df[df["delta_value"] > cutoff]["z_noise"].dropna().to_numpy(dtype=float)
    cliffs = _cliffs_delta(near, rest)
    plt.figure(figsize=(6, 5))
    if near.size == 0 and rest.size == 0:
        plt.text(0.5, 0.5, "No data available after filtering", ha="center", va="center")
        plt.axis("off")
    else:
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

    # Family-specific delta histogram (real vs null)
    plot_family_hist_delta(df, family_hist, output_dir, rng)

    # Scatter delta vs z_noise (delta on x)
    plot_delta_vs_znoise(df, output_dir)

    # Loss curves for selected families (using Fm_corr)
    if loss_families is None:
        loss_families = ["SC_Binary", "SC_HighPressure"]
    plot_loss_curves(df, loss_families, output_dir, lambda_penalty=lambda_penalty, bounds=bounds)

    # Fig S01: Noise (z_noise) by family
    if "z_noise" in df.columns:
        df_noise = df.copy()
        fam_label_map = {
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
        df_noise["family_label"] = df_noise["category"].apply(lambda x: fam_label_map.get(x, str(x)))
        order_noise = df_noise.groupby("family_label")["z_noise"].median().sort_values().index.tolist()
        data_noise = [df_noise.loc[df_noise["family_label"] == fam, "z_noise"].dropna().to_numpy() for fam in order_noise]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data_noise, tick_labels=order_noise, whis=[5, 95], showfliers=False, patch_artist=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Z-score noise")
        plt.title("Structural noise by family (ordered by median)")
        figS01 = output_dir / "figS01_noise_by_family.png"
        plt.tight_layout()
        plt.savefig(figS01, dpi=200)
        plt.close()

    # Fig S02: Tc vs Tc_ideal scatter
    if "Tc_K" in merged.columns and "Tc_ideal" in merged.columns:
        tc_df = merged.copy()
        tc_df["Tc_K"] = pd.to_numeric(tc_df["Tc_K"], errors="coerce")
        tc_df["Tc_ideal"] = pd.to_numeric(tc_df["Tc_ideal"], errors="coerce")
        tc_df = tc_df.dropna(subset=["Tc_K", "Tc_ideal"])
        if not tc_df.empty:
            plt.figure(figsize=(6, 6))
            plt.scatter(tc_df["Tc_K"], tc_df["Tc_ideal"], alpha=0.5, s=18, color="#2b8cbe")
            lim = max(tc_df["Tc_K"].max(), tc_df["Tc_ideal"].max()) * 1.05
            plt.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="Tc_ideal = Tc")
            plt.xlim(0, lim)
            plt.ylim(0, lim)
            plt.xlabel("Tc (K)")
            plt.ylabel("Tc_ideal = Tc * (1 + predicted_noise) (K)")
            plt.title("Tc vs Tc_ideal")
            plt.legend()
            figS02 = output_dir / "figS02_Tc_vs_Tc_ideal.png"
            plt.tight_layout()
            plt.savefig(figS02, dpi=200)
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
    parser.add_argument("--family-hist", type=str, default="SC_Binary", help="Family to plot for real vs null delta histogram")
    parser.add_argument("--loss-families", nargs="*", default=None, help="Families to include in loss curves")
    parser.add_argument("--lambda-penalty", type=float, default=0.001, help="Penalty weight for loss curves")
    parser.add_argument("--bounds", nargs=2, type=float, default=(0.5, 5.0), help="Bounds for base-frequency grid in loss curves")
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
        family_hist=args.family_hist,
        loss_families=args.loss_families,
        lambda_penalty=args.lambda_penalty,
        bounds=tuple(args.bounds),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
