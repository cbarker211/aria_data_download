import numpy as np
from haversine import haversine

def build_grid(site_lat, site_lon, source, dellat=0.06, dellon=0.08):

    # Define extent around target scene to sample
    # (balance between encompassing the scene and amount of data permitted for 10m resolution):

    loc_lat  = float(site_lat)
    loc_lon  = float(site_lon)
    lon_min = loc_lon - dellon
    lat_min = loc_lat - dellat
    lon_max = loc_lon + dellon
    lat_max = loc_lat + dellat

    # Calculate box size in km²
    lat_km = haversine((lat_min, loc_lon), (lat_max, loc_lon))
    lon_km = haversine((loc_lat, lon_min), (loc_lat, lon_max))
    print(f"Bounding box size: {lat_km:.2f} km x {lon_km:.2f} km")
    bbox = [lon_min, lat_min, lon_max, lat_max] # [west, south, east, north]

    lat_range = np.array([lat_min, lat_max], dtype=np.float64)  # [lat_min, lat_max]
    lon_range = np.array([lon_min, lon_max], dtype=np.float64)  # [lon_min, lon_max]

    return bbox, lat_range, lon_range