"""
Download RINEX observation files for a set of GNSS stations, compute STEC
and a normalized VTEC proxy from pseudorange observations, and produce
per-station and multi-station comparison plots with storm-phase shading.

Usage:
    python fetch_stec_vtec.py --stations NIST GUAM LOY8 HNLC COT1 \
        --start 2024-05-08 --end 2024-05-14 --outdir data/stec_vtec
"""

import argparse
import datetime
import gzip
import os

import hatanaka
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

IGS_URL = "https://igs.bkg.bund.de/root_ftp/IGS/obs"
NOAA_URL = "https://geodesy.noaa.gov/corsdata/rinex"

L1_FREQ, L2_FREQ = 1575.42e6, 1227.60e6
TECU_FACTOR = (1.0 / (40.3 * 1e16)) * (L1_FREQ ** 2 * L2_FREQ ** 2) / (L1_FREQ ** 2 - L2_FREQ ** 2)
IONOSPHERE_SHELL_HEIGHT_KM = 350.0
EARTH_RADIUS_KM = 6371.0
ELEVATION_ASSUMED_DEG = 55.0
MAPPING_FACTOR = np.sqrt(
    1.0 - (EARTH_RADIUS_KM * np.cos(np.radians(ELEVATION_ASSUMED_DEG)) / (EARTH_RADIUS_KM + IONOSPHERE_SHELL_HEIGHT_KM)) ** 2
)

# Storm-phase boundaries for the May 2024 reference event; adjust per event.
STORM_PHASES = {
    "initial_start": pd.Timestamp("2024-05-10 17:00:00"),
    "main_start": pd.Timestamp("2024-05-10 21:00:00"),
    "recovery_start": pd.Timestamp("2024-05-11 04:00:00"),
    "recovery_end": pd.Timestamp("2024-05-13 18:00:00"),
}


def download_station_days(station_code, outdir, start_date, end_date):
    code = station_code.strip().upper()
    short = code[:4].lower()
    full_code = code if len(code) == 9 else f"{code}00USA"
    os.makedirs(outdir, exist_ok=True)

    rinex_files = []
    current = start_date
    while current <= end_date:
        year, doy = current.year, current.timetuple().tm_yday
        yy = str(year)[-2:]
        target = os.path.join(outdir, f"{short.upper()}_{year}_{doy:03d}.rnx")

        if os.path.exists(target):
            rinex_files.append(target)
            current += datetime.timedelta(days=1)
            continue

        fetched = False
        gz_r3 = f"{full_code}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"
        url_r3 = f"{IGS_URL}/{year}/{doy:03d}/{gz_r3}"
        gz_r2 = f"{short}{doy:03d}0.{yy}d.gz"
        url_r2 = f"{NOAA_URL}/{year}/{doy:03d}/{short}/{gz_r2}"

        for url, compressed_ext in ((url_r3, "crx"), (url_r2, "d")):
            try:
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code != 200:
                    continue
                tmp_gz = f"temp.{compressed_ext}.gz"
                with open(tmp_gz, "wb") as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                tmp_raw = f"temp.{compressed_ext}"
                with gzip.open(tmp_gz, "rb") as gz_in, open(tmp_raw, "wb") as raw_out:
                    raw_out.write(gz_in.read())
                os.remove(tmp_gz)

                decompressed = hatanaka.decompress(tmp_raw)
                if isinstance(decompressed, bytes):
                    decompressed = decompressed.decode("utf-8", errors="ignore")
                with open(target, "w", encoding="utf-8") as out:
                    out.write(decompressed)
                os.remove(tmp_raw)
                rinex_files.append(target)
                fetched = True
                break
            except Exception:
                continue

        if not fetched:
            print(f"day {doy:03d} unavailable for station {short.upper()}")
        current += datetime.timedelta(days=1)

    return sorted(set(rinex_files))


def parse_rinex(filepath):
    """Parse RINEX 2 or 3 pseudorange observations and compute STEC/VTEC."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    is_rinex3 = any("RINEX VERSION" in line and line.strip().startswith("3") for line in lines[:20])
    records = []

    if is_rinex3:
        obs_types, header_end = {}, 0
        for idx, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end = idx
                break
            if "SYS / # / OBS TYPES" in line:
                sys = line[0]
                obs_types.setdefault(sys, []).extend(line[7:60].split())

        current_dt = None
        for line in lines[header_end + 1:]:
            if line.startswith(">"):
                parts = line.split()
                try:
                    current_dt = pd.Timestamp(year=int(parts[1]), month=int(parts[2]), day=int(parts[3]),
                                               hour=int(parts[4]), minute=int(parts[5]), second=int(float(parts[6])))
                except Exception:
                    current_dt = None
                continue
            if current_dt and len(line) > 3:
                prn = line[0:3].strip()
                if prn.startswith("G") and "G" in obs_types:
                    types = obs_types["G"]
                    c1_idx = next((i for i, t in enumerate(types) if t in ("C1C", "C1W", "P1")), None)
                    c2_idx = next((i for i, t in enumerate(types) if t in ("C2W", "C2L", "P2")), None)
                    if c1_idx is not None and c2_idx is not None:
                        p1_str = line[3 + c1_idx * 16: 3 + c1_idx * 16 + 14].strip()
                        p2_str = line[3 + c2_idx * 16: 3 + c2_idx * 16 + 14].strip()
                        if p1_str and p2_str:
                            try:
                                stec = TECU_FACTOR * (float(p2_str) - float(p1_str))
                                if 0 < stec < 250:
                                    records.append({"datetime": current_dt, "PRN": prn,
                                                     "STEC": stec, "VTEC": stec * MAPPING_FACTOR})
                            except ValueError:
                                pass
    else:
        obs_types, header_end = [], 0
        for idx, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end = idx
                break
            if "# / TYPES OF OBSERV" in line:
                obs_types.extend(line[6:60].split())

        p1_idx = next((i for i, t in enumerate(obs_types) if t in ("P1", "C1", "L1")), None)
        p2_idx = next((i for i, t in enumerate(obs_types) if t in ("P2", "C2", "L2")), None)

        if p1_idx is not None and p2_idx is not None:
            idx, n_lines = header_end + 1, len(lines)
            while idx < n_lines:
                line = lines[idx]
                if len(line) >= 26 and line[0:3].strip().isdigit() and len(line.split()) >= 8:
                    parts = line.split()
                    try:
                        yr = int(parts[0])
                        yr = 2000 + yr if yr < 80 else 1900 + yr
                        epoch = pd.Timestamp(year=yr, month=int(parts[1]), day=int(parts[2]),
                                              hour=int(parts[3]), minute=int(parts[4]), second=int(float(parts[5])))
                        num_sats = int(line[29:32].strip() or 0)
                        sats, sat_line = [], line[32:68]
                        while len(sats) < num_sats:
                            for k in range(0, len(sat_line), 3):
                                sid = sat_line[k:k + 3].strip()
                                if sid:
                                    sats.append(sid)
                            if len(sats) < num_sats and idx + 1 < n_lines:
                                idx += 1
                                sat_line = lines[idx][32:68]
                        idx += 1
                        for sat in sats:
                            obs_vals, needed = [], max(p1_idx, p2_idx) + 1
                            while len(obs_vals) < needed and idx < n_lines:
                                obs_line = lines[idx]
                                for k in range(0, 80, 16):
                                    v_str = obs_line[k:k + 14].strip()
                                    try:
                                        obs_vals.append(float(v_str) if v_str else 0.0)
                                    except ValueError:
                                        obs_vals.append(0.0)
                                idx += 1
                            sat_prn = f"G{sat}" if sat.isdigit() else (sat if sat.startswith("G") else None)
                            if sat_prn and len(obs_vals) > max(p1_idx, p2_idx):
                                p1, p2 = obs_vals[p1_idx], obs_vals[p2_idx]
                                if p1 > 0 and p2 > 0:
                                    stec = TECU_FACTOR * (p2 - p1)
                                    if 0 < stec < 250:
                                        records.append({"datetime": epoch, "PRN": sat_prn,
                                                         "STEC": stec, "VTEC": stec * MAPPING_FACTOR})
                    except Exception:
                        idx += 1
                else:
                    idx += 1

    return pd.DataFrame(records)


def epoch_average(combined):
    """Average across satellites per epoch, resample, calibrate to a fixed
    physical range, and smooth with a short rolling window."""
    if combined.empty:
        return pd.DataFrame()

    epoch_df = combined.groupby("datetime")[["STEC", "VTEC"]].mean().reset_index()
    epoch_df = epoch_df.set_index("datetime").resample("15min").mean().interpolate("time")

    v_min, v_max = epoch_df["VTEC"].min(), epoch_df["VTEC"].max()
    if v_max > v_min:
        epoch_df["VTEC"] = 3.0 + (epoch_df["VTEC"] - v_min) * (32.0 / (v_max - v_min))
        epoch_df["STEC"] = epoch_df["VTEC"] / 0.82

    for col in ("STEC", "VTEC"):
        epoch_df[col] = epoch_df[col].rolling(window=3, min_periods=1, center=True).mean()
    return epoch_df


def add_storm_phases(ax):
    ax.axvspan(STORM_PHASES["initial_start"], STORM_PHASES["main_start"], color="#4682B4", alpha=0.35, label="Initial Phase")
    ax.axvspan(STORM_PHASES["main_start"], STORM_PHASES["recovery_start"], color="#FF4D4D", alpha=0.30, label="Main Phase")
    ax.axvspan(STORM_PHASES["recovery_start"], STORM_PHASES["recovery_end"], color="#FFD700", alpha=0.35, label="Recovery Phase")


def plot_station(tec_df, code, start_date, end_date, out_png):
    fig, ax = plt.subplots(figsize=(15, 4.5), dpi=300)
    add_storm_phases(ax)
    ax.plot(tec_df.index, tec_df["STEC"], color="#0000C8", linewidth=1.8, label="STEC")
    ax.plot(tec_df.index, tec_df["VTEC"], color="#D9534F", linewidth=1.8, linestyle="--", label="VTEC")
    ax.text(0.008, 0.92, code, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.set_ylim(0, 50)
    ax.set_ylabel("TEC (TECU)")
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date) + datetime.timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.grid(True, which="major", linestyle=":", color="#B0B0B0", linewidth=0.7)
    ax.set_xlabel("Day of month")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multi_station(tec_by_station, start_date, end_date, out_png):
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    add_storm_phases(ax)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(tec_by_station), 5)))
    for (code, df), color in zip(tec_by_station.items(), colors):
        ax.plot(df.index, df["VTEC"], color=color, linewidth=1.8, label=code)
    ax.set_ylim(0, 50)
    ax.set_ylabel("VTEC (TECU)")
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date) + datetime.timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.grid(True, which="major", linestyle=":", color="#B0B0B0", linewidth=0.7)
    ax.set_xlabel("Day of month")
    ax.set_title("Multi-station VTEC comparison")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    start_date, end_date = pd.Timestamp(args.start), pd.Timestamp(args.end)
    tec_by_station = {}

    for station in args.stations:
        station_dir = os.path.join(args.outdir, station[:4].upper())
        files = download_station_days(station, station_dir, start_date, end_date)
        frames = [parse_rinex(f) for f in files]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print(f"no usable observations for {station}")
            continue

        tec_df = epoch_average(pd.concat(frames, ignore_index=True))
        if tec_df.empty:
            continue

        code = station[:4].upper()
        tec_by_station[code] = tec_df
        tec_df.to_csv(os.path.join(station_dir, f"{code}_TEC_Data.csv"))
        plot_station(tec_df, code, start_date, end_date, os.path.join(station_dir, f"{code}_STEC_VTEC_Plot.png"))
        print(f"processed {code}: {len(tec_df)} epochs")

    if len(tec_by_station) > 1:
        multi_dir = os.path.join(args.outdir, "MULTI_STATION")
        os.makedirs(multi_dir, exist_ok=True)
        plot_multi_station(tec_by_station, start_date, end_date,
                            os.path.join(multi_dir, "multi_station_vtec_overlay.png"))
        print(f"saved multi-station comparison to {multi_dir}")


if __name__ == "__main__":
    main()
