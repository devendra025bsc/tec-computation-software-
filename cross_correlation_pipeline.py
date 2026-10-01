"""
Full analysis pipeline: builds the master station-day table across all
events and station pairs, computes lagged cross-correlation with surrogate
significance testing and block-bootstrap confidence intervals (on both the
peak correlation and the peak lag), classifies storm phase from SYM-H, and
applies Benjamini-Hochberg FDR correction across all valid tests.

This is the pipeline that produced the numerical results reported in the
manuscript; the standalone scripts in scripts/ were used only to collect
the raw input data.

Usage:
    python analysis/cross_correlation_pipeline.py --config config.json --outdir results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

LAG_RANGE_MIN = range(-180, 181, 5)
RESAMPLE_MINUTES = 5
N_SURROGATES = 500
N_BOOTSTRAP = 500
N_LAG_BOOTSTRAP = 200
MIN_SHIFT_MINUTES = 30
MIN_VALID_SAMPLES = 30
QUIET_SYMH_THRESHOLD_NT = -50


def load_geomagnetic(path):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    return df.set_index("Timestamp")[["dB_dt"]]


def load_roti(path):
    df = pd.read_csv(path, parse_dates=["Time"])
    return df.set_index("Time")["ROTI"]


def load_symh(path):
    df = pd.read_csv(path, parse_dates=["Time"])
    df["Time"] = df["Time"].dt.tz_localize(None)
    return df.set_index("Time")["SYM_H"]


def classify_storm_phase(symh_daily_min):
    """Classify Quiet / Main Phase / Recovery from the daily-minimum SYM-H
    series of a single event, independently of any correlation results."""
    ordered = symh_daily_min.sort_index()
    min_day = ordered.idxmin()
    phases = {}
    reached_minimum = False
    for day, value in ordered.items():
        if day < min_day:
            phases[day] = "Main Phase" if value < QUIET_SYMH_THRESHOLD_NT else "Quiet"
        elif day == min_day:
            phases[day] = "Main Phase"
        else:
            phases[day] = "Recovery"
    return phases


def prepare_day(dbdt_series, roti_series, day):
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    dbdt_day = dbdt_series.loc[(dbdt_series.index >= day_start) & (dbdt_series.index < day_end), "dB_dt"]
    dbdt_resampled = dbdt_day.resample(f"{RESAMPLE_MINUTES}min").mean()
    roti_day = roti_series[(roti_series.index >= day_start) & (roti_series.index < day_end)]
    combined = pd.concat([dbdt_resampled.rename("dbdt"), roti_day.rename("roti")], axis=1).dropna()
    return combined


def ccf_scan(dbdt, roti, n, lag_range=LAG_RANGE_MIN, step=RESAMPLE_MINUTES):
    """CCF(tau) = corr[dB/dt(t), ROTI(t+tau)]. Positive tau: dB/dt leads."""
    results = {}
    for lag in lag_range:
        shift = lag // step
        if shift >= 0:
            a = dbdt[: n - shift] if shift > 0 else dbdt
            b = roti[shift:]
        else:
            s = -shift
            a, b = dbdt[s:], roti[: n - s]
        if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
            continue
        results[lag] = np.corrcoef(a, b)[0, 1]
    return results


def peak_lag(results):
    lag = max(results, key=lambda k: abs(results[k]))
    return lag, results[lag]


def surrogate_pvalue(dbdt, roti, n, observed_abs_r, n_surrogates=N_SURROGATES,
                      min_shift_min=MIN_SHIFT_MINUTES, rng=None):
    """Circular-shift surrogate test: shift ROTI by a random offset and
    recompute the maximum |r| across the full lag range. This preserves the
    autocorrelation structure of each series while destroying any true
    phase relationship."""
    rng = rng or np.random.default_rng()
    min_shift = min_shift_min // RESAMPLE_MINUTES
    count_ge = 0

    for _ in range(n_surrogates):
        shift = rng.integers(min_shift, max(min_shift + 1, n - min_shift))
        roti_shifted = np.roll(roti, shift)
        best = 0.0
        for lag in LAG_RANGE_MIN:
            step_shift = lag // RESAMPLE_MINUTES
            if step_shift >= 0:
                a = dbdt[: n - step_shift] if step_shift > 0 else dbdt
                b = roti_shifted[step_shift:]
            else:
                s = -step_shift
                a, b = dbdt[s:], roti_shifted[: n - s]
            if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
                continue
            r = abs(np.corrcoef(a, b)[0, 1])
            best = max(best, r)
        if best >= observed_abs_r:
            count_ge += 1

    return (count_ge + 1) / (n_surrogates + 1)


def block_bootstrap_r_ci(dbdt, roti, n, lag, n_bootstrap=N_BOOTSTRAP,
                          block_min=MIN_SHIFT_MINUTES, ci=95, rng=None):
    """Block-bootstrap 95% CI for the correlation at a fixed lag."""
    rng = rng or np.random.default_rng()
    step_shift = lag // RESAMPLE_MINUTES
    if step_shift >= 0:
        a = dbdt[: n - step_shift] if step_shift > 0 else dbdt
        b = roti[step_shift:]
    else:
        s = -step_shift
        a, b = dbdt[s:], roti[: n - s]

    m = len(a)
    block_len = max(1, block_min // RESAMPLE_MINUTES)
    n_blocks = int(np.ceil(m / block_len))
    boot_r = []

    for _ in range(n_bootstrap):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, m - block_len))
            idx.extend(range(start, min(start + block_len, m)))
        idx = np.array(idx[:m])
        aa, bb = a[idx], b[idx]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        boot_r.append(np.corrcoef(aa, bb)[0, 1])

    lo, hi = np.percentile(boot_r, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


def block_bootstrap_lag_ci(dbdt, roti, n, n_bootstrap=N_LAG_BOOTSTRAP,
                            block_min=MIN_SHIFT_MINUTES, ci=95, rng=None):
    """Block-bootstrap 95% CI for the peak lag itself, by re-running the
    full lag scan on each resample and taking the percentile of the
    resulting argmax distribution."""
    rng = rng or np.random.default_rng()
    block_len = max(1, block_min // RESAMPLE_MINUTES)
    n_blocks = int(np.ceil(n / block_len))
    lags_found = []

    for _ in range(n_bootstrap):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n - block_len))
            idx.extend(range(start, min(start + block_len, n)))
        idx = np.array(idx[:n])
        d_b, r_b = dbdt[idx], roti[idx]

        best_r, best_lag = 0, 0
        for lag in LAG_RANGE_MIN:
            step_shift = lag // RESAMPLE_MINUTES
            if step_shift >= 0:
                a = d_b[: n - step_shift] if step_shift > 0 else d_b
                b = r_b[step_shift:]
            else:
                s = -step_shift
                a, b = d_b[s:], r_b[: n - s]
            if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
                continue
            r = np.corrcoef(a, b)[0, 1]
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        lags_found.append(best_lag)

    lags_found = np.array(lags_found)
    lo, hi = np.percentile(lags_found, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi, np.std(lags_found)


def run_station_day(dbdt_series, roti_series, day, rng=None):
    combined = prepare_day(dbdt_series, roti_series, day)
    if len(combined) < MIN_VALID_SAMPLES:
        return None

    dbdt, roti, n = combined["dbdt"].values, combined["roti"].values, len(combined)
    results = ccf_scan(dbdt, roti, n)
    if not results:
        return None

    lag, r = peak_lag(results)
    p_value = surrogate_pvalue(dbdt, roti, n, abs(r), rng=rng)
    r_ci_lo, r_ci_hi = block_bootstrap_r_ci(dbdt, roti, n, lag, rng=rng)
    lag_ci_lo, lag_ci_hi, lag_sd = block_bootstrap_lag_ci(dbdt, roti, n, rng=rng)

    return {
        "peak_r": r, "peak_lag_min": lag, "p_value": p_value,
        "r_ci_lower": r_ci_lo, "r_ci_upper": r_ci_hi,
        "lag_ci_lower": lag_ci_lo, "lag_ci_upper": lag_ci_hi, "lag_bootstrap_sd": lag_sd,
        "n_samples": n, "max_abs_dbdt": np.nanmax(np.abs(dbdt)), "max_roti": np.nanmax(roti),
    }


def build_master_table(config, rng=None):
    """config: dict of {event_label: {"days": [...], "stations": {station_label:
    {"dbdt_path": ..., "roti_path": ..., "symh_path": ..., "mag_lat": ...}}}}"""
    rng = rng or np.random.default_rng(42)
    rows = []

    for event_label, event_cfg in config.items():
        days = pd.to_datetime(event_cfg["days"])

        symh_daily_min = {}
        for station_label, station_cfg in event_cfg["stations"].items():
            symh = load_symh(station_cfg["symh_path"])
            symh_daily_min = symh.resample("D").min().to_dict()
            break
        storm_phase = classify_storm_phase(pd.Series(symh_daily_min))

        for station_label, station_cfg in event_cfg["stations"].items():
            dbdt_series = load_geomagnetic(station_cfg["dbdt_path"])
            roti_series = load_roti(station_cfg["roti_path"])

            for day in days:
                result = run_station_day(dbdt_series, roti_series, day, rng=rng)
                day_str = day.strftime("%Y-%m-%d")
                row = {
                    "event": event_label, "station_pair": station_label, "date": day_str,
                    "storm_phase": storm_phase.get(pd.Timestamp(day_str), "N/A"),
                    "mag_lat_deg": station_cfg.get("mag_lat"),
                }
                if result is None:
                    row.update({k: np.nan for k in
                                ["peak_r", "peak_lag_min", "p_value", "r_ci_lower", "r_ci_upper",
                                 "lag_ci_lower", "lag_ci_upper", "lag_bootstrap_sd", "max_abs_dbdt", "max_roti"]})
                else:
                    row.update(result)
                rows.append(row)

    df = pd.DataFrame(rows)

    valid = df.dropna(subset=["peak_r"]).copy()
    if len(valid) > 0:
        rejected, p_adjusted, _, _ = multipletests(valid["p_value"].values, alpha=0.05, method="fdr_bh")
        valid["p_fdr"] = p_adjusted
        valid["significant_fdr"] = np.where(rejected, "Yes", "No")
        valid["significant_nominal"] = np.where(valid["p_value"] < 0.05, "Yes", "No")
        df = df.merge(valid[["event", "station_pair", "date", "p_fdr", "significant_fdr", "significant_nominal"]],
                       on=["event", "station_pair", "date"], how="left")

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON file describing events/stations/file paths")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    table = build_master_table(config)
    out_path = os.path.join(args.outdir, "master_table.csv")
    table.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(table)} rows, {table['peak_r'].notna().sum()} valid)")


if __name__ == "__main__":
    main()
