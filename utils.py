import numpy as np
from io import StringIO
import pandas as pd
from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS

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
    else:
        raise ImportError(f"Failed to fetch from JSR URL: {url}", response.status_code)
    return df

def build_grid(site_lat, site_lon):

    # Define extent around target scene to sample
    # (balance between encompassing the scene and amount of data permitted for 10m resolution):

    dellat, dellon = 0.05, 0.08
    loc_lat  = float(site_lat)
    loc_lon  = float(site_lon)
    lllon = loc_lon - dellon
    lllat = loc_lat - dellat
    urlon = loc_lon + dellon
    urlat = loc_lat + dellat
    coords_wgs84 = (lllon, lllat, urlon, urlat)
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    lat_range = np.array([lllat, urlat], dtype=np.float64)  # [lat_min, lat_max]
    lon_range = np.array([lllon, urlon], dtype=np.float64)  # [lon_min, lon_max]

    return bbox, lat_range, lon_range