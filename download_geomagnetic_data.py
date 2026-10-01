"""
Download 1-minute geomagnetic variation data (X, Y, Z) from the USGS
geomagnetism web service and compute |dB/dt| for a given observatory
and date range.

Usage:
    python download_geomagnetic_data.py --station BOU \
        --start 2024-05-08 --end 2024-05-15 --outdir data/geomag
"""
import argparse, os
import numpy as np
import pandas as pd
import requests

USGS_URL = "https://geomag.usgs.gov/ws/data/"
FILL_THRESHOLD = 88880


def fetch_iaga2002(station, start, end):
    params = {
        "id": station,
        "starttime": start.strftime("%Y-%m-%dT00:00:00Z"),
        "endtime": end.strftime("%Y-%m-%dT00:00:00Z"),
        "elements": "X,Y,Z", "sampling_period": "60",
        "type": "variation", "format": "iaga2002",
    }
    r = requests.get(USGS_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def parse_iaga2002(text):
    skip = ("#", "Format", "Source", "Station", "DATE", "Generated", "Reported", "Data")
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(skip):
            continue
        p = line.split()
        if len(p) < 6:
            continue
        try:
            ts = pd.to_datetime(f"{p[0]} {p[1]}", utc=True)
            records.append([ts, float(p[3]), float(p[4]), float(p[5])])
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(records, columns=["Timestamp", "X", "Y", "Z"])
    return df.sort_values("Timestamp").drop_duplicates("Timestamp").set_index("Timestamp")


def clean(df, start, end):
    for c in ("X", "Y", "Z"):
        df.loc[df[c].abs() >= FILL_THRESHOLD, c] = np.nan
    df = df[(df.index >= start) & (df.index < end)].copy()
    idx = pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
    return df.reindex(idx)


def add_derivatives(df):
    for c in ("X", "Y", "Z"):
        prev, nxt = df[c].shift(1), df[c].shift(-1)
        valid = df[c].notna() & prev.notna() & nxt.notna()
        d = (nxt - prev) / 2.0
        d.loc[~valid] = np.nan
        df[f"d{c}_dt"] = d
    df["dB_H_dt"] = np.sqrt(df["dX_dt"]**2 + df["dY_dt"]**2)
    df["dB_dt"]   = np.sqrt(df["dX_dt"]**2 + df["dY_dt"]**2 + df["dZ_dt"]**2)
    return df


def add_dst(df):
    df["Dst"] = np.nan
    try:
        r = requests.get(
            "https://services.swpc.noaa.gov/json/geospace/geospace_dst_1h.json",
            timeout=15)
        r.raise_for_status()
        dst = pd.DataFrame(r.json())
        dst["Timestamp"] = pd.to_datetime(dst["time_tag"], utc=True)
        dst = dst.set_index("Timestamp")[["dst"]].astype(float)
        df["Dst"] = dst.reindex(df.index).interpolate(method="time")["dst"]
    except Exception:
        baseline = df["X"].iloc[:1440].mean()
        df["Dst"] = df["X"] - baseline
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    stn   = a.station.strip().upper()
    start = pd.Timestamp(a.start, tz="UTC")
    end   = pd.Timestamp(a.end,   tz="UTC")

    text = fetch_iaga2002(stn, start, end)
    df   = parse_iaga2002(text)
    df   = clean(df, start, end)
    df   = add_derivatives(df)
    df   = add_dst(df)

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{stn}_Geomagnetic_Data_{start.strftime('%Y%m%d')}.csv")
    df.reset_index().to_csv(out, index=False)
    print(f"saved {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
