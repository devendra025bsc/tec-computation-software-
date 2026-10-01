"""
Generate all manuscript figures from the master results table.

Produces Figures 1-18 used in the paper, in publication-quality PNG and PDF
(300 dpi, serif font, Okabe-Ito colorblind-safe palette).

Usage:
    python src/generate_figures.py \
        --master-csv results/master_table.csv \
        --omni-csv data/omni/omni_data_20240508_20240514.csv \
        --outdir figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

# ── publication-quality style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.edgecolor": "#999999",
    "legend.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
      "#56B4E9", "#F0E442", "#000000"]

EVENT_ORDER = ["Aug 2018 (G3)", "Nov 2021 (G3)", "Nov 2023 (G3)", "Mar 2024 (G4)", "May 2024 (G5)"]
STATION_ORDER = ["BOU+NIST", "COT1+TUC", "GUA+GUAM", "HON+HNLC", "LOY8+FRD"]
MARKERS = ["o", "s", "^", "D", "v"]


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")


# ── Fig 1 – station map ───────────────────────────────────────────────────────
def fig_station_map(out_dir):
    stations = {
        "BOU+NIST": (40.13, -105.24, 49.5),
        "COT1+TUC": (32.17, -110.73, 39.9),
        "GUA+GUAM": (13.59, 144.87,   5.0),
        "HON+HNLC": (21.32, -157.87, 21.7),
        "LOY8+FRD": (38.20, -77.37,  48.4),
    }
    label_offsets = {
        "BOU+NIST": (8, 14), "COT1+TUC": (8, -18), "GUA+GUAM": (8, 6),
        "HON+HNLC": (8, 6),  "LOY8+FRD": (8, -6),
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_facecolor("#dbe9f6")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="white", lw=0.4)
    for lat in range(-60, 76, 15):
        ax.axhline(lat, color="white", lw=0.4)
    ax.axhline(0, color="#888888", lw=0.6, ls="--")
    ax.axvline(0, color="#888888", lw=0.6, ls="--")

    for (name, (lat, lon, maglat)), c in zip(stations.items(), OI):
        ax.scatter(lon, lat, color=c, s=140, edgecolor="black", linewidth=1.0, zorder=5)
        dx, dy = label_offsets[name]
        ax.annotate(f"{name}\n({lat:.1f}°N, {lon:.1f}°)",
                    xy=(lon, lat), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", zorder=6)

    ax.set_xlabel("Geographic longitude (°E)")
    ax.set_ylabel("Geographic latitude (°N)")
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-60, 76, 15))
    ax.set_title("Global distribution of the five geomagnetic–GNSS station pairs")
    ax.grid(False)
    save(fig, out_dir, "Fig1_station_map")


# ── Fig 6 – optimal-lag heatmap ───────────────────────────────────────────────
def fig_lag_heatmap(df, out_dir):
    piv = df.groupby(["station_pair", "event"])["peak_lag_min"].mean().unstack()
    piv = piv.reindex(index=STATION_ORDER, columns=EVENT_ORDER)
    vmax = np.nanmax(np.abs(piv.values))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(EVENT_ORDER)))
    ax.set_xticklabels(EVENT_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(STATION_ORDER)))
    ax.set_yticklabels(STATION_ORDER)
    for i in range(len(STATION_ORDER)):
        for j in range(len(EVENT_ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.55 else "black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean peak lag τ (min)\n(+: |dB/dt| leads, −: ROTI leads)")
    ax.set_title("Station-pair × event optimal-lag heatmap (mean τ over valid days)")
    ax.grid(False)
    save(fig, out_dir, "Fig6_lag_heatmap")


# ── Fig 7 – correlation heatmap ───────────────────────────────────────────────
def fig_corr_heatmap(df, out_dir):
    piv = df.groupby(["station_pair", "event"])["peak_r"].mean().unstack()
    piv = piv.reindex(index=STATION_ORDER, columns=EVENT_ORDER)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(EVENT_ORDER)))
    ax.set_xticklabels(EVENT_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(STATION_ORDER)))
    ax.set_yticklabels(STATION_ORDER)
    for i in range(len(STATION_ORDER)):
        for j in range(len(EVENT_ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.22 else "black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean peak correlation r")
    ax.set_title("Station-pair × event peak-correlation heatmap (mean r over valid days)")
    ax.grid(False)
    save(fig, out_dir, "Fig7_correlation_heatmap")


# ── Fig 8 – lag vs magnetic latitude ─────────────────────────────────────────
def fig_lag_vs_maglat(df, out_dir):
    v = df.dropna(subset=["peak_lag_min", "mag_lat_deg"])
    rho, p = stats.spearmanr(v["mag_lat_deg"], v["peak_lag_min"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ev, c, m in zip(EVENT_ORDER, OI, MARKERS):
        sub = v[v["event"] == ev]
        jitter = np.random.default_rng(42).uniform(-0.5, 0.5, len(sub))
        ax.scatter(sub["mag_lat_deg"] + jitter, sub["peak_lag_min"],
                   label=ev, color=c, marker=m, s=38, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xlabel("Approximate geomagnetic latitude (°N)")
    ax.set_ylabel("Peak lag τ (min)")
    ax.set_title(f"Peak lag vs. magnetic latitude  (Spearman ρ={rho:.2f}, p={p:.3f}, n={len(v)})")
    ax.legend(fontsize=7.5)
    save(fig, out_dir, "Fig8_lag_vs_maglat")


# ── Fig 9 – |r| vs magnetic latitude ─────────────────────────────────────────
def fig_r_vs_maglat(df, out_dir):
    v = df.dropna(subset=["peak_r", "mag_lat_deg"])
    rho, p = stats.spearmanr(v["mag_lat_deg"], v["peak_r"].abs())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ev, c, m in zip(EVENT_ORDER, OI, MARKERS):
        sub = v[v["event"] == ev]
        jitter = np.random.default_rng(42).uniform(-0.5, 0.5, len(sub))
        ax.scatter(sub["mag_lat_deg"] + jitter, sub["peak_r"].abs(),
                   label=ev, color=c, marker=m, s=38, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Approximate geomagnetic latitude (°N)")
    ax.set_ylabel("|Peak correlation r|")
    ax.set_title(f"|Peak r| vs. magnetic latitude  (Spearman ρ={rho:.2f}, p={p:.3f}, n={len(v)})")
    ax.legend(fontsize=7.5)
    save(fig, out_dir, "Fig9_r_vs_maglat")


# ── Fig 11 – lag vs MLT (polar) ───────────────────────────────────────────────
def fig_lag_vs_mlt(df, out_dir):
    v = df.dropna(subset=["peak_lag_min", "mlt_at_symh_min_hr"])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    theta = v["mlt_at_symh_min_hr"].values / 24.0 * 2 * np.pi
    r_abs = v["peak_lag_min"].abs().values
    colors = np.where(v["peak_lag_min"].values >= 0, OI[0], OI[1])
    ax.scatter(theta, r_abs, c=colors, s=35, alpha=0.75,
               edgecolor="white", linewidth=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["00", "03", "06", "09", "12", "15", "18", "21"])
    ax.set_ylabel("|Peak lag| (min)", labelpad=30)
    ax.set_title("Peak lag vs. MLT at SYM-H minimum\n(blue: |dB/dt| leads; orange: ROTI leads)",
                 fontsize=10.5, pad=20)

    handles = [
        mpatches.Patch(color=OI[0], label="|dB/dt| leads (τ > 0)"),
        mpatches.Patch(color=OI[1], label="ROTI leads (τ < 0)"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=8)
    save(fig, out_dir, "Fig11_lag_vs_mlt_polar")


# ── Fig 12 – lag vs SYM-H min ────────────────────────────────────────────────
def fig_lag_vs_symh(df, out_dir):
    v = df.dropna(subset=["peak_lag_min", "sym_h_min_nt"])
    rho, p = stats.spearmanr(v["sym_h_min_nt"].abs(), v["peak_lag_min"].abs())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ev, c, m in zip(EVENT_ORDER, OI, MARKERS):
        sub = v[v["event"] == ev]
        ax.scatter(sub["sym_h_min_nt"], sub["peak_lag_min"],
                   label=ev, color=c, marker=m, s=38, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xlabel("Daily minimum SYM-H (nT)")
    ax.set_ylabel("Peak lag τ (min)")
    ax.set_title(f"Peak lag vs. minimum SYM-H  (Spearman ρ={rho:.2f} on |values|, p={p:.3f}, n={len(v)})")
    ax.legend(fontsize=7.5)
    save(fig, out_dir, "Fig12_lag_vs_symh")


# ── Fig 13 – storm-phase comparison ──────────────────────────────────────────
def fig_phase_comparison(df, out_dir):
    phase_order = ["Quiet", "Main Phase", "Recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    data_r = [df.loc[df["storm_phase"] == p, "peak_r"].abs().dropna().values for p in phase_order]
    bp = axes[0].boxplot(data_r, tick_labels=phase_order, patch_artist=True,
                          showmeans=True,
                          medianprops=dict(color="#D55E00", lw=1.6),
                          meanprops=dict(marker="D", markerfacecolor="white",
                                         markeredgecolor="black"))
    for patch, c in zip(bp["boxes"], ["#56B4E9", "#E69F00", "#009E73"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    h, p = stats.kruskal(*data_r)
    axes[0].set_title(f"(a) |Peak r| by storm phase\nKruskal-Wallis H={h:.2f}, p={p:.3f}")
    axes[0].set_ylabel("|Peak correlation r|")

    data_lag = [df.loc[df["storm_phase"] == p, "peak_lag_min"].dropna().values for p in phase_order]
    bp2 = axes[1].boxplot(data_lag, tick_labels=phase_order, patch_artist=True,
                           showmeans=True,
                           medianprops=dict(color="#D55E00", lw=1.6),
                           meanprops=dict(marker="D", markerfacecolor="white",
                                          markeredgecolor="black"))
    for patch, c in zip(bp2["boxes"], ["#56B4E9", "#E69F00", "#009E73"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    h2, p2 = stats.kruskal(*data_lag)
    axes[1].axhline(0, color="#999999", lw=0.7)
    axes[1].set_title(f"(b) Peak lag by storm phase\nKruskal-Wallis H={h2:.2f}, p={p2:.3f}")
    axes[1].set_ylabel("Peak lag τ (min)")

    fig.suptitle("Storm-phase comparison (Quiet / Main / Recovery)", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "Fig13_phase_comparison")


# ── Fig 13b – |r| vs max |dB/dt| ─────────────────────────────────────────────
def fig_r_vs_dbdt(df, out_dir):
    v = df.dropna(subset=["peak_r", "max_abs_dbdt"])
    rho, p = stats.spearmanr(v["max_abs_dbdt"], v["peak_r"].abs())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ev, c, m in zip(EVENT_ORDER, OI, MARKERS):
        sub = v[v["event"] == ev]
        ax.scatter(sub["max_abs_dbdt"], sub["peak_r"].abs(),
                   label=ev, color=c, marker=m, s=38, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Maximum |dB/dt| (nT/min)")
    ax.set_ylabel("|Peak correlation r|")
    ax.set_title(f"|Peak r| vs. maximum |dB/dt|  (Spearman ρ={rho:.2f}, p={p:.3f}, n={len(v)})")
    ax.legend(fontsize=7.5)
    save(fig, out_dir, "Fig13b_r_vs_dbdt")


# ── Fig 14 – cross-event summary ─────────────────────────────────────────────
def fig_cross_event_summary(df, out_dir):
    valid = df.dropna(subset=["peak_r"])
    event_stats = (
        valid.groupby("event").agg(
            pct_sig=("significant_nominal", lambda x: 100 * (x == "Yes").mean()),
            median_r=("peak_r", lambda x: x.abs().median()),
            iqr_r=("peak_r", lambda x: x.abs().quantile(0.75) - x.abs().quantile(0.25)),
            max_r=("peak_r", lambda x: x.abs().max()),
        )
        .reindex(EVENT_ORDER)
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    axes[0].bar(event_stats.index, event_stats["pct_sig"], color=OI[0], edgecolor="black", lw=0.5)
    axes[0].set_title("(a) % significant (nominal p<0.05)")
    axes[0].set_ylabel("% of valid days")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, max(20, event_stats["pct_sig"].max() * 1.3))

    axes[1].bar(event_stats.index, event_stats["median_r"],
                yerr=event_stats["iqr_r"] / 2, capsize=4,
                color=OI[1], edgecolor="black", lw=0.5)
    axes[1].set_title("(b) Median |peak r| (± ½IQR)")
    axes[1].set_ylabel("|Peak r|")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(event_stats.index, event_stats["max_r"], color=OI[2], edgecolor="black", lw=0.5)
    axes[2].set_title("(c) Strongest |peak r| observed")
    axes[2].set_ylabel("Max |peak r|")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Cross-event summary statistics", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "Fig14_cross_event_summary")


# ── Fig 15 – propagation / separation ────────────────────────────────────────
def fig_propagation_separation(df, out_dir):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = (np.sin(dphi / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlam / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(a))

    station_coords = {
        "BOU+NIST": (40.13, -105.24), "COT1+TUC": (32.17, -110.73),
        "GUA+GUAM": (13.59, 144.87),  "HON+HNLC": (21.32, -157.87),
        "LOY8+FRD": (38.20, -77.37),
    }
    stations = list(station_coords.keys())
    pairs, lon_diffs, distances, lag_diffs = [], [], [], []

    for ev in EVENT_ORDER:
        sub = df[df["event"] == ev].dropna(subset=["peak_lag_min"])
        mean_lag = sub.groupby("station_pair")["peak_lag_min"].mean()
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                s1, s2 = stations[i], stations[j]
                if s1 not in mean_lag or s2 not in mean_lag:
                    continue
                lat1, lon1 = station_coords[s1]
                lat2, lon2 = station_coords[s2]
                lon_diff = abs(lon1 - lon2)
                lon_diff = min(lon_diff, 360 - lon_diff)
                dist = haversine(lat1, lon1, lat2, lon2)
                lag_diff = abs(mean_lag[s1] - mean_lag[s2])
                lon_diffs.append(lon_diff)
                distances.append(dist)
                lag_diffs.append(lag_diff)
                pairs.append((ev, s1, s2))

    lon_diffs = np.array(lon_diffs)
    distances = np.array(distances)
    lag_diffs = np.array(lag_diffs)
    rho1, p1 = stats.spearmanr(lon_diffs, lag_diffs)
    rho2, p2 = stats.spearmanr(distances, lag_diffs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(lon_diffs, lag_diffs, color=OI[0], s=35, alpha=0.8, edgecolor="white", lw=0.3)
    axes[0].set_xlabel("Longitude difference (°)")
    axes[0].set_ylabel("|Inter-station lag difference| (min)")
    axes[0].set_title(f"(a) vs. longitude separation\nSpearman ρ={rho1:.3f}, p={p1:.2f}")

    axes[1].scatter(distances, lag_diffs, color=OI[1], s=35, alpha=0.8, edgecolor="white", lw=0.3)
    axes[1].set_xlabel("Great-circle distance (km)")
    axes[1].set_ylabel("|Inter-station lag difference| (min)")
    axes[1].set_title(f"(b) vs. great-circle distance\nSpearman ρ={rho2:.3f}, p={p2:.2f}")

    fig.suptitle("Inter-station-pair lag difference vs. spatial separation", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "Fig15_propagation_separation")


# ── Fig 16 – quiet vs storm control ──────────────────────────────────────────
def fig_quiet_vs_storm(df, out_dir):
    quiet = df[df["storm_phase"] == "Quiet"]["peak_r"].abs().dropna()
    storm = df[df["storm_phase"].isin(["Main Phase", "Recovery"])]["peak_r"].abs().dropna()
    u, p = stats.mannwhitneyu(storm, quiet, alternative="two-sided")
    rb = 1 - (2 * u) / (len(storm) * len(quiet))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bp = ax.boxplot([quiet, storm], tick_labels=["Quiet\n(control)", "Storm\n(Main+Recovery)"],
                    patch_artist=True, showmeans=True,
                    medianprops=dict(color="#D55E00", lw=1.6),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    for patch, c in zip(bp["boxes"], ["#56B4E9", "#E69F00"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("|Peak correlation r|")
    ax.set_title(f"Quiet-time control vs. storm-time\nMann-Whitney p={p:.4f}, rank-biserial={rb:.3f}")
    save(fig, out_dir, "Fig16_quiet_vs_storm_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--omni-csv", default=None)
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    df = pd.read_csv(args.master_csv)

    # normalise column names (allow underscores or hyphens, upper or lower)
    df.columns = [c.lower().replace("-", "_") for c in df.columns]

    for col_alias, canonical in [
        ("max_abs_dBdt_nT_per_min", "max_abs_dbdt"),
        ("max_roti_tecu_per_min", "max_roti"),
        ("sym_h_min_nt", "sym_h_min_nt"),
        ("mlt_at_symh_min_hr", "mlt_at_symh_min_hr"),
    ]:
        src = col_alias.lower().replace("-", "_")
        if src in df.columns and canonical not in df.columns:
            df[canonical] = df[src]

    np.random.seed(42)

    fig_station_map(args.outdir)
    fig_lag_heatmap(df, args.outdir)
    fig_corr_heatmap(df, args.outdir)
    fig_lag_vs_maglat(df, args.outdir)
    fig_r_vs_maglat(df, args.outdir)
    fig_phase_comparison(df, args.outdir)
    fig_r_vs_dbdt(df, args.outdir)
    fig_cross_event_summary(df, args.outdir)
    fig_propagation_separation(df, args.outdir)
    fig_quiet_vs_storm(df, args.outdir)

    if "mlt_at_symh_min_hr" in df.columns:
        fig_lag_vs_mlt(df, args.outdir)
    if "sym_h_min_nt" in df.columns:
        fig_lag_vs_symh(df, args.outdir)

    print("all figures saved.")


if __name__ == "__main__":
    main()
