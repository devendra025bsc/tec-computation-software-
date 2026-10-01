"""
Download daily RINEX observation files for a GNSS station from the NOAA
CORS archive and compute ROTI (Rate Of TEC change Index) from dual-frequency
carrier-phase observations.

ROTI is defined following Pi et al. (1997) as the standard deviation of the
rate of TEC change (ROT) over a short sliding window.

Usage:
    python compute_roti_from_rinex.py --station nist \
        --start 2024-05-08 --end 2024-05-15 --outdir data/roti
"""

import argparse
import gzip
import os
import shutil
import urllib.request

import georinex as gr
import hatanaka
import numpy as np
import pandas as pd

BASE_URLS = [
    "https://noaa-cors-pds.s3.amazonaws.com/rinex",
    "https://geodesy.noaa.gov/corsdata/rinex",
]

L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6
SPEED_OF_LIGHT = 299792458.0


def download_station_days(station_id, outdir, start_date, end_date):
    os.makedirs(outdir, exist_ok=True)
    date_range = pd.date_range(start_date, end_date, freq="D")
    downloaded = []

    for date in date_range:
        year, yr_short, doy = date.year, date.strftime("%y"), date.timetuple().tm_yday
        out_o = os.path.join(outdir, f"{station_id}{doy:03d}0.{yr_short}o")
        if os.path.exists(out_o):
            downloaded.append(out_o)
            continue

        fetched = False
        for ext in (f"0.{yr_short}d.gz", f"0.{yr_short}o.gz"):
            if fetched:
                break
            filename = f"{station_id}{doy:03d}{ext}"
            gz_path = os.path.join(outdir, filename)
            for base_url in BASE_URLS:
                url = f"{base_url}/{year}/{doy:03d}/{station_id}/{filename}"
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request) as response, open(gz_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)

                    if filename.endswith("d.gz"):
                        d_path = os.path.join(outdir, f"{station_id}{doy:03d}0.{yr_short}d")
                        with gzip.open(gz_path, "rb") as f_in, open(d_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(gz_path)
                        o_path = hatanaka.decompress_on_disk(d_path)
                        if os.path.exists(d_path) and o_path != d_path:
                            os.remove(d_path)
                        downloaded.append(o_path)
                    else:
                        with gzip.open(gz_path, "rb") as f_in, open(out_o, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(gz_path)
                        downloaded.append(out_o)
                    fetched = True
                    break
                except Exception:
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
            if fetched:
                break
        if not fetched:
            print(f"day {doy:03d} unavailable for station {station_id}")

    return downloaded


def stec_from_carrier_phase(rinex_file):
    """Extract L1/L2 carrier phase from a RINEX file and compute per-satellite
    STEC, rate-of-TEC (ROT), and rolling ROTI."""
    if not os.path.exists(rinex_file):
        return pd.DataFrame()

    try:
        obs = gr.load(rinex_file)
    except Exception as exc:
        print(f"could not parse {rinex_file}: {exc}")
        return pd.DataFrame()

    if obs is None or not hasattr(obs, "data_vars"):
        return pd.DataFrame()

    available = list(obs.data_vars.keys())
    l1_var = next((v for v in ("L1", "L1C", "LA", "L1P", "L1X") if v in available), None)
    l2_var = next((v for v in ("L2", "L2C", "LB", "L2P", "L2W", "L2X") if v in available), None)
    if not l1_var or not l2_var:
        return pd.DataFrame()

    df = obs[[l1_var, l2_var]].to_dataframe().dropna().reset_index()
    df = df.rename(columns={l1_var: "L1", l2_var: "L2"})
    for col in ("sv", "satellite", "PRN"):
        if col in df.columns:
            df = df.rename(columns={col: "PRN"})
    for col in ("time", "Epoch", "Time"):
        if col in df.columns:
            df = df.rename(columns={col: "time"})

    df = df[df["PRN"].astype(str).str.startswith("G")].copy()
    if df.empty:
        return pd.DataFrame()

    lam1, lam2 = SPEED_OF_LIGHT / L1_FREQ, SPEED_OF_LIGHT / L2_FREQ
    k_factor = (1.0 / (40.3 * 1e16)) * (L1_FREQ ** 2 * L2_FREQ ** 2) / (L1_FREQ ** 2 - L2_FREQ ** 2)

    df["STEC"] = k_factor * (df["L1"] * lam1 - df["L2"] * lam2)
    df = df.sort_values(["PRN", "time"])

    df["dt_min"] = df.groupby("PRN")["time"].diff().dt.total_seconds() / 60.0
    df["dSTEC"] = df.groupby("PRN")["STEC"].diff()
    df["ROT"] = df["dSTEC"] / df["dt_min"]

    valid = (df["dt_min"].between(0.1, 2.0)) & (df["dSTEC"].abs() < 15.0) & (df["ROT"].abs() < 30.0)
    df = df[valid].copy()
    df["ROTI_sat"] = df.groupby("PRN")["ROT"].transform(lambda x: x.rolling(10, min_periods=4).std())

    return df.groupby("time")["ROTI_sat"].mean().reset_index().rename(
        columns={"time": "Time", "ROTI_sat": "ROTI"}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True, help="4-character CORS station ID")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    station = args.station.strip().lower()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    station_dir = os.path.join(args.outdir, station)

    files = download_station_days(station, station_dir, start_date, end_date)

    frames = []
    for f in files:
        result = stec_from_carrier_phase(f)
        if not result.empty:
            frames.append(result)

    if not frames:
        raise RuntimeError(f"no ROTI could be computed for station {station}; check RINEX availability")

    roti = pd.concat(frames).sort_values("Time").drop_duplicates("Time")
    roti = roti.set_index("Time").resample("5min").mean().reset_index()

    out_path = os.path.join(station_dir, f"processed_roti_{station}_{start_date.strftime('%Y%m%d')}.csv")
    roti.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(roti)} rows)")


if __name__ == "__main__":
    main()"""
Download daily RINEX observation files for a GNSS station from the NOAA
CORS archive and compute ROTI (Rate Of TEC change Index) from dual-frequency
carrier-phase observations.

ROTI is defined following Pi et al. (1997) as the standard deviation of the
rate of TEC change (ROT) over a short sliding window.

Usage:
    python compute_roti_from_rinex.py --station nist \
        --start 2024-05-08 --end 2024-05-15 --outdir data/roti
"""

import argparse
import gzip
import os
import shutil
import urllib.request

import georinex as gr
import hatanaka
import numpy as np
import pandas as pd

BASE_URLS = [
    "https://noaa-cors-pds.s3.amazonaws.com/rinex",
    "https://geodesy.noaa.gov/corsdata/rinex",
]

L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6
SPEED_OF_LIGHT = 299792458.0


def download_station_days(station_id, outdir, start_date, end_date):
    os.makedirs(outdir, exist_ok=True)
    date_range = pd.date_range(start_date, end_date, freq="D")
    downloaded = []

    for date in date_range:
        year, yr_short, doy = date.year, date.strftime("%y"), date.timetuple().tm_yday
        out_o = os.path.join(outdir, f"{station_id}{doy:03d}0.{yr_short}o")
        if os.path.exists(out_o):
            downloaded.append(out_o)
            continue

        fetched = False
        for ext in (f"0.{yr_short}d.gz", f"0.{yr_short}o.gz"):
            if fetched:
                break
            filename = f"{station_id}{doy:03d}{ext}"
            gz_path = os.path.join(outdir, filename)
            for base_url in BASE_URLS:
                url = f"{base_url}/{year}/{doy:03d}/{station_id}/{filename}"
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request) as response, open(gz_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)

                    if filename.endswith("d.gz"):
                        d_path = os.path.join(outdir, f"{station_id}{doy:03d}0.{yr_short}d")
                        with gzip.open(gz_path, "rb") as f_in, open(d_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(gz_path)
                        o_path = hatanaka.decompress_on_disk(d_path)
                        if os.path.exists(d_path) and o_path != d_path:
                            os.remove(d_path)
                        downloaded.append(o_path)
                    else:
                        with gzip.open(gz_path, "rb") as f_in, open(out_o, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        os.remove(gz_path)
                        downloaded.append(out_o)
                    fetched = True
                    break
                except Exception:
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
            if fetched:
                break
        if not fetched:
            print(f"day {doy:03d} unavailable for station {station_id}")

    return downloaded


def stec_from_carrier_phase(rinex_file):
    """Extract L1/L2 carrier phase from a RINEX file and compute per-satellite
    STEC, rate-of-TEC (ROT), and rolling ROTI."""
    if not os.path.exists(rinex_file):
        return pd.DataFrame()

    try:
        obs = gr.load(rinex_file)
    except Exception as exc:
        print(f"could not parse {rinex_file}: {exc}")
        return pd.DataFrame()

    if obs is None or not hasattr(obs, "data_vars"):
        return pd.DataFrame()

    available = list(obs.data_vars.keys())
    l1_var = next((v for v in ("L1", "L1C", "LA", "L1P", "L1X") if v in available), None)
    l2_var = next((v for v in ("L2", "L2C", "LB", "L2P", "L2W", "L2X") if v in available), None)
    if not l1_var or not l2_var:
        return pd.DataFrame()

    df = obs[[l1_var, l2_var]].to_dataframe().dropna().reset_index()
    df = df.rename(columns={l1_var: "L1", l2_var: "L2"})
    for col in ("sv", "satellite", "PRN"):
        if col in df.columns:
            df = df.rename(columns={col: "PRN"})
    for col in ("time", "Epoch", "Time"):
        if col in df.columns:
            df = df.rename(columns={col: "time"})

    df = df[df["PRN"].astype(str).str.startswith("G")].copy()
    if df.empty:
        return pd.DataFrame()

    lam1, lam2 = SPEED_OF_LIGHT / L1_FREQ, SPEED_OF_LIGHT / L2_FREQ
    k_factor = (1.0 / (40.3 * 1e16)) * (L1_FREQ ** 2 * L2_FREQ ** 2) / (L1_FREQ ** 2 - L2_FREQ ** 2)

    df["STEC"] = k_factor * (df["L1"] * lam1 - df["L2"] * lam2)
    df = df.sort_values(["PRN", "time"])

    df["dt_min"] = df.groupby("PRN")["time"].diff().dt.total_seconds() / 60.0
    df["dSTEC"] = df.groupby("PRN")["STEC"].diff()
    df["ROT"] = df["dSTEC"] / df["dt_min"]

    valid = (df["dt_min"].between(0.1, 2.0)) & (df["dSTEC"].abs() < 15.0) & (df["ROT"].abs() < 30.0)
    df = df[valid].copy()
    df["ROTI_sat"] = df.groupby("PRN")["ROT"].transform(lambda x: x.rolling(10, min_periods=4).std())

    return df.groupby("time")["ROTI_sat"].mean().reset_index().rename(
        columns={"time": "Time", "ROTI_sat": "ROTI"}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True, help="4-character CORS station ID")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    station = args.station.strip().lower()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    station_dir = os.path.join(args.outdir, station)

    files = download_station_days(station, station_dir, start_date, end_date)

    frames = []
    for f in files:
        result = stec_from_carrier_phase(f)
        if not result.empty:
            frames.append(result)

    if not frames:
        raise RuntimeError(f"no ROTI could be computed for station {station}; check RINEX availability")

    roti = pd.concat(frames).sort_values("Time").drop_duplicates("Time")
    roti = roti.set_index("Time").resample("5min").mean().reset_index()

    out_path = os.path.join(station_dir, f"processed_roti_{station}_{start_date.strftime('%Y%m%d')}.csv")
    roti.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(roti)} rows)")


if __name__ == "__main__":
    main()
