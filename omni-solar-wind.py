"""
Download hourly OMNI solar-wind and geomagnetic-index data (IMF Bz, solar
wind speed and velocity components, Kp, Dst, AE) from NASA OMNIWeb for a
given date range, and produce a multi-panel overview plot.

Usage:
    python fetch_omni_solar_wind.py --start 20240508 --end 20240515 --outdir data/omni
"""

import argparse
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests

OMNIWEB_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# OMNIWeb variable codes: 16=Bz(GSM), 24=Vsw, 21/22/23=Vx/Vy/Vz, 38=Kp*10, 40=Dst, 41=AE
OMNI_VARS = ["16", "24", "21", "22", "23", "38", "40", "41"]
COLUMN_NAMES = ["Year", "DOY", "Hour", "Bz_GSM", "Vsw", "Vx", "Vy", "Vz", "Kp", "Dst", "AE"]

FILL_LIMITS = {
    "Bz_GSM": 500, "Vsw": 5000, "Vx": 5000, "Vy": 5000, "Vz": 5000,
    "Kp": 90, "Dst": 1000, "AE": 5000,
}


def fetch_omni(start_date, end_date):
    payload = [
        ("activity", "retrieve"), ("res", "hour"), ("spacecraft", "omni2"),
        ("start_date", start_date), ("end_date", end_date),
    ] + [("vars", v) for v in OMNI_VARS]

    response = requests.post(OMNIWEB_URL, data=payload, timeout=60)
    response.raise_for_status()

    data_lines = [line for line in response.text.splitlines() if line.strip()[:4].isdigit()]
    if not data_lines:
        raise RuntimeError(f"no OMNI data returned for {start_date}-{end_date}")

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s+", header=None, names=COLUMN_NAMES)
    for col in COLUMN_NAMES[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, limit in FILL_LIMITS.items():
        df[col] = df[col].mask(df[col].abs() > limit)

    df["Time"] = pd.to_datetime(df["Year"].astype(str) + df["DOY"].astype(str).str.zfill(3),
                                 format="%Y%j") + pd.to_timedelta(df["Hour"], unit="h")
    return df.set_index("Time")


def plot_omni(df, start_date, end_date, out_png):
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(df.index, df["Bz_GSM"], color="red")
    axes[0].set_ylabel("IMF Bz (nT)")
    axes[0].set_title(f"Solar-wind and geomagnetic-index overview ({start_date}-{end_date})")

    axes[1].plot(df.index, df["Vsw"], color="green")
    axes[1].set_ylabel("Vsw (km/s)")

    axes[2].plot(df.index, df["Vx"], label="Vx", color="blue")
    axes[2].plot(df.index, df["Vy"], label="Vy", color="orange")
    axes[2].plot(df.index, df["Vz"], label="Vz", color="purple")
    axes[2].set_ylabel("Velocity (km/s)")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(df.index, df["Kp"], color="magenta")
    axes[3].set_ylabel("Kp x 10")

    axes[4].plot(df.index, df["Dst"], color="black")
    axes[4].set_ylabel("Dst (nT)")

    axes[5].plot(df.index, df["AE"], color="brown")
    axes[5].set_ylabel("AE (nT)")
    axes[5].set_xlabel("UTC")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = fetch_omni(args.start, args.end)

    csv_path = os.path.join(args.outdir, f"omni_data_{args.start}_{args.end}.csv")
    df.to_csv(csv_path)
    print(f"saved {csv_path} ({len(df)} rows)")

    png_path = os.path.join(args.outdir, f"omni_overview_{args.start}_{args.end}.png")
    plot_omni(df, args.start, args.end, png_path)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
