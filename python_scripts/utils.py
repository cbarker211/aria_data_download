import numpy as np
from io import StringIO
import pandas as pd
from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS
from haversine import haversine

def scrape_jsr(url,session):
    # Function to web scrape data and convert to a pandas DataFrame.
    response = session.get(url)
    if response.status_code == 200:

        filename = url.split("/")[-1]  # Use the last part of the URL as filename
        with open(f"gcat/{filename}", "w", encoding="utf-8") as f:
            f.write(response.text)

        # Convert the content to a file-like object for pandas
        tsv_data = StringIO(response.text)
        # Load the data into a pandas DataFrame
        df = pd.read_csv(tsv_data, delimiter="\t", dtype=object, low_memory=False)
        df = df.iloc[1:].reset_index(drop=True)
    else:
        raise ImportError(f"Failed to fetch from JSR URL: {url}", response.status_code)
    return df

def build_grid(site_lat, site_lon):

    # Define extent around target scene to sample
    # (balance between encompassing the scene and amount of data permitted for 10m resolution):

    dellat, dellon = 0.02, 0.02
    loc_lat  = float(site_lat)
    loc_lon  = float(site_lon)
    lon_min = loc_lon - dellon
    lat_min = loc_lat - dellat
    lon_max = loc_lon + dellon
    lat_max = loc_lat + dellat
    coords_wgs84 = (lon_min, lat_min, lon_max, lat_max)
    bbox_gee = [lon_min, lat_min, lon_max, lat_max]  # [west, south, east, north]
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    lat_range = np.array([lat_min, lat_max], dtype=np.float64)  # [lat_min, lat_max]
    lon_range = np.array([lon_min, lon_max], dtype=np.float64)  # [lon_min, lon_max]

    # Calculate box size in km²
    lat_km = haversine((lat_min, loc_lon), (lat_max, loc_lon))
    lon_km = haversine((loc_lat, lon_min), (loc_lat, lon_max))
    print(f"Bounding box size: {lat_km:.2f} km x {lon_km:.2f} km")

    return bbox, lat_range, lon_range, bbox_gee