import datetime
import os
import requests
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

def fetch_usgs_geomagnetic_data(station: str, start_dt: datetime.datetime, end_dt: datetime.datetime) -> pd.DataFrame:
    """
    Downloads raw IAGA-2002 geomagnetic data from the USGS web service
    for a specific date/time range and parses it into a cleaned DataFrame.
    """
    STATION = station.strip().upper()
    START = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    END = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    start_time = pd.Timestamp(start_dt).tz_localize("UTC") if start_dt.tzinfo is None else pd.Timestamp(start_dt)
    end_time   = pd.Timestamp(end_dt).tz_localize("UTC") if end_dt.tzinfo is None else pd.Timestamp(end_dt)

    URL = "https://geomag.usgs.gov/ws/data/"
    PARAMS = {
        "id": STATION,
        "starttime": START,
        "endtime": END,
        "elements": "X,Y,Z",
        "sampling_period": "60",
        "type": "variation",
        "format": "iaga2002",
    }

    response = requests.get(URL, params=PARAMS, timeout=60)
    response.raise_for_status()
    text = response.text

    if len(text.strip()) == 0:
        raise RuntimeError(f"USGS returned an empty response for station code '{STATION}'.")

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "Format", "Source", "Station", "DATE", "Generated", "Reported", "Data")):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            timestamp = pd.to_datetime(f"{parts[0]} {parts[1]}", utc=True, errors="raise")
            X = float(parts[3])
            Y = float(parts[4])
            Z = float(parts[5])
            records.append([timestamp, X, Y, Z])
        except (ValueError, TypeError):
            continue

    if not records:
        raise RuntimeError(f"No valid IAGA-2002 observations parsed for station '{STATION}'.")

    df = pd.DataFrame(records, columns=["Timestamp", "X", "Y", "Z"])
    df = df.sort_values("Timestamp").drop_duplicates(subset="Timestamp", keep="first").set_index("Timestamp")

    for component in ["X", "Y", "Z"]:
        df.loc[df[component].abs() >= 88880, component] = np.nan

    df = df[(df.index >= start_time) & (df.index < end_time)].copy()
    full_index = pd.date_range(start=start_time, end=end_time - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "Timestamp"

    return df


# ================================================================
# CROSS-VERSION COMPATIBLE INTERACTIVE UI
# ================================================================

style = {'description_width': '90px'}

# Options for hours (00-23) and minutes (00-59)
hours_list = [(f"{h:02d}", h) for h in range(24)]
mins_list = [(f"{m:02d}", m) for m in range(60)]

station_widget = widgets.Text(
    value='BOU',
    description='Station:',
    placeholder='e.g. BOU, HON, FRD',
    style=style
)

# Start Date & Time controls
start_date_widget = widgets.DatePicker(
    description='Start Date:',
    value=datetime.date(2024, 5, 8),
    style=style
)
start_hour_widget = widgets.Dropdown(options=hours_list, value=0, description='Hour UTC:', layout=widgets.Layout(width='150px'))
start_min_widget = widgets.Dropdown(options=mins_list, value=0, description='Min:', layout=widgets.Layout(width='120px'))

# End Date & Time controls
end_date_widget = widgets.DatePicker(
    description='End Date:',
    value=datetime.date(2024, 5, 15),
    style=style
)
end_hour_widget = widgets.Dropdown(options=hours_list, value=0, description='Hour UTC:', layout=widgets.Layout(width='150px'))
end_min_widget = widgets.Dropdown(options=mins_list, value=0, description='Min:', layout=widgets.Layout(width='120px'))

download_button = widgets.Button(
    description="Download CSV",
    button_style='success',
    icon='download'
)

output = widgets.Output()


def on_download_clicked(b):
    with output:
        clear_output(wait=True)
        try:
            # Construct start and end datetimes
            s_time = datetime.time(start_hour_widget.value, start_min_widget.value)
            start_dt = datetime.datetime.combine(start_date_widget.value, s_time)

            e_time = datetime.time(end_hour_widget.value, end_min_widget.value)
            end_dt = datetime.datetime.combine(end_date_widget.value, e_time)

            station = station_widget.value.strip().upper()

            if start_dt >= end_dt:
                print("Error: Start datetime must be before End datetime.")
                return

            print(f"Fetching data for {station} from {start_dt} to {end_dt} UTC...")
            df = fetch_usgs_geomagnetic_data(station, start_dt, end_dt)

            filename = f"{station}_geomagnetic_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename)
            print(f"File successfully generated: {filename}")

            # Trigger automatic browser download in Google Colab
            try:
                from google.colab import files
                files.download(filename)
                print("Download triggered in browser.")
            except ImportError:
                print(f"Saved locally to: {os.path.abspath(filename)}")

        except Exception as e:
            print(f"An error occurred: {e}")


download_button.on_click(on_download_clicked)

# Render UI layout
ui = widgets.VBox([
    widgets.HTML("<h3>USGS Geomagnetic Data Downloader</h3>"),
    widgets.HBox([station_widget]),
    widgets.HBox([start_date_widget, start_hour_widget, start_min_widget]),
    widgets.HBox([end_date_widget, end_hour_widget, end_min_widget]),
    widgets.Box([download_button], layout=widgets.Layout(margin='10px 0px 10px 0px')),
    output
])

display(ui)
