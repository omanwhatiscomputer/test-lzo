"""
Plots and summary numbers for rank_stats.csv files written by rank_probe.py.

Usage:
    python analyze_rank.py result/*/rank_stats.csv --out rank_plots
    python analyze_rank.py run.csv --kappa 1.0     # also report r*(k) (optional extension)

Per run it writes:
    <run>_ranks.png     r_eff, r_part, r_delta, gap rank vs step (median over layers, IQR band),
                        lozo vs null (and grad_proj if present), with the training loss overlaid
    <run>_spectrum.png  normalized singular value spectrum at 10/25/50/75/100% of training
    <run>_layers.png    heatmap of r_eff (lozo) per layer x interval
and one summary.csv with one row per (run, source).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["r_eff", "r_part", "r_0.01", "r_0.05", "r_0.10", "gap_rank"]
SOURCES = ["lozo", "null", "grad_proj", "grad_full"]
COLORS = {"lozo": "#2a6fdb", "null": "#999999", "grad_proj": "#d9480f", "grad_full": "#2b8a3e"}


def sv_cols(df):
    return [c for c in df.columns if c.startswith("sv_")]


def optimal_rank(svs, kappa):
    """r*(k) = argmax_r' sqrt(sum_{i<=r'} s_i^2) / sqrt(kappa * r' + 1)."""
    s = np.sort(svs[~np.isnan(svs)])[::-1]
    if len(s) == 0:
        return np.nan
    num = np.sqrt(np.cumsum(s ** 2))
    den = np.sqrt(kappa * np.arange(1, len(s) + 1) + 1)
    return int(np.argmax(num / den) + 1)


def trend(x, y):
    """Spearman correlation between interval index and a metric (monotone-trend test)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.unique(y[ok]).size < 2 or np.unique(x[ok]).size < 2:
        return np.nan
    return pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())


def plot_run(df, name, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    loss = df[df.source == "lozo"].groupby("step")["loss"].first()
    for ax, m in zip(axes.flat, METRICS):
        for src in ["lozo", "null", "grad_proj"]:
            d = df[df.source == src]
            if d.empty:
                continue
            g = d.groupby("step")[m]
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med.values, color=COLORS[src], label=src, lw=1.5)
            ax.fill_between(med.index, lo.values, hi.values, color=COLORS[src], alpha=0.15, lw=0)
        ax.set_title(m)
        ax.set_ylim(0, df.rank_r.iloc[0] + 0.5)
        ax.grid(alpha=0.3)
        if not loss.empty:
            ax2 = ax.twinx()
            ax2.plot(loss.index, loss.values, color="black", alpha=0.25, lw=1)
            ax2.set_yticks([])
    axes[0, 0].legend(loc="lower left", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(f"{name}  (median over layers, IQR band; faint black = train loss)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_ranks.png"), dpi=120)
    plt.close(fig)

    # Spectrum snapshots
    d = df[df.source == "lozo"]
    steps = np.sort(d.step.unique())
    if len(steps) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        cols = sv_cols(d)[: int(df.rank_r.iloc[0])]
        for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            st = steps[min(len(steps) - 1, max(0, int(round(frac * len(steps))) - 1))]
            S = d[d.step == st][cols].to_numpy(dtype=float)
            S = S / S[:, :1]
            ax.plot(np.arange(1, len(cols) + 1), np.nanmedian(S, 0), marker="o", label=f"{int(frac * 100)}% (step {st})")
        ns = df[df.source == "null"][cols].to_numpy(dtype=float)
        if len(ns):
            ax.plot(np.arange(1, len(cols) + 1), np.nanmedian(ns / ns[:, :1], 0), "k--", label="null (all steps)")
        ax.set_xlabel("i")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$ (median over layers)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_spectrum.png"), dpi=120)
        plt.close(fig)

        # Per-layer heatmap
        piv = d.pivot_table(index="layer", columns="interval", values="r_eff")
        fig, ax = plt.subplots(figsize=(12, max(4, 0.08 * len(piv))))
        im = ax.imshow(piv.to_numpy(), aspect="auto", interpolation="nearest", vmin=1, vmax=df.rank_r.iloc[0])
        ax.set_xlabel("interval k")
        ax.set_ylabel("layer")
        if len(piv) <= 60:
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels(piv.index, fontsize=5)
        fig.colorbar(im, ax=ax, label="r_eff (lozo)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_layers.png"), dpi=120)
        plt.close(fig)


def summarize(df, name, kappa):
    rows = []
    for src in SOURCES:
        d = df[df.source == src]
        if d.empty:
            continue
        per_int = d.groupby("interval")[METRICS].median()
        k = per_int.index.to_numpy(dtype=float)
        n = len(per_int)
        early, late = per_int.iloc[: max(1, n // 10)], per_int.iloc[-max(1, n // 10):]
        row = {"run": name, "source": src, "rank_r": d.rank_r.iloc[0], "nu": d.step_interval.iloc[0], "intervals": n}
        for m in ["r_eff", "r_part", "r_0.05", "gap_rank"]:
            row[f"{m}_mean"] = per_int[m].mean()
            row[f"{m}_early"] = early[m].mean()
            row[f"{m}_late"] = late[m].mean()
            row[f"{m}_trend_rho"] = trend(k, per_int[m].to_numpy(dtype=float))
        # spread across layers (time-averaged), for per-layer rank selection
        per_layer = d.groupby("layer")["r_eff"].mean()
        row["r_eff_layer_min"], row["r_eff_layer_max"] = per_layer.min(), per_layer.max()
        if src == "grad_proj" and "energy_frac" in d:
            row["energy_frac_mean"] = pd.to_numeric(d.energy_frac, errors="coerce").mean()
        if kappa is not None:
            S = d[sv_cols(d)[: int(d.rank_r.iloc[0])]].to_numpy(dtype=float)
            row["r_star_mean"] = np.nanmean([optimal_rank(s, kappa) for s in S])
        rows.append(row)
    # signal check: how far is lozo from the sign-randomized null?
    lz, nl = df[df.source == "lozo"], df[df.source == "null"]
    if len(lz) and len(nl):
        for r in rows:
            r["fro2_lozo_over_null"] = lz.fro2.mean() / nl.fro2.mean()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="rank_plots")
    ap.add_argument("--kappa", type=float, default=None, help="constant in r*(k) = argmax sqrt(sum s^2)/sqrt(kappa r' + 1)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    summary = []
    for f in a.files:
        df = pd.read_csv(f, keep_default_na=False, na_values=[""])  # "null" is a source name, not NaN
        name = os.path.basename(os.path.dirname(os.path.abspath(f))) or os.path.splitext(os.path.basename(f))[0]
        plot_run(df, name, a.out)
        summary += summarize(df, name, a.kappa)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s[["run", "source", "rank_r", "nu", "intervals", "r_eff_mean", "r_eff_early", "r_eff_late",
                 "r_eff_trend_rho", "r_eff_layer_min", "r_eff_layer_max"]
                + [c for c in ["fro2_lozo_over_null", "energy_frac_mean", "r_star_mean"] if c in s]].round(3).to_string())


if __name__ == "__main__":
    main()
