import ee
from datetime import datetime, timedelta, timezone
import numpy as np

def search_gee(launch_data, launch_data_dict, landsat, region, scene, cloud_cover, site):

    # Set up the time range.
    launch_time = launch_data_dict["launch_time"]
    time_start =  (launch_time + timedelta(hours=0)).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end   =  (launch_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Search Landsat 8/9 Collection 2 Level-2 (Surface Reflectance)
    landsat_filtered = landsat.filterBounds(region) \
                           .filterDate(time_start, time_end) \
                           .filter(ee.Filter.lt('CLOUD_COVER', 30))
    
    if landsat_filtered.size().getInfo() == 0:
        return launch_data

    # Isolate the image.
    landsat_image = landsat_filtered.first()

    # Get metadata.
    time_ms = landsat_image.get('system:time_start').getInfo()
    overpass_dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
    satellite = landsat_image.get('SPACECRAFT_ID').getInfo()

    # Select specific bands.
    bands = ["QA_PIXEL"]  
    landsat_image = landsat_image.select(bands)
    data = landsat_image.sampleRectangle(region=region, defaultValue=0).getInfo()
    
    arrays = []
    for band in bands:
        arrays.append(np.array(data['properties'][band]))

    qa = np.stack(arrays, axis=0) #(band, y, x)

    def extract_bits(arr, start, end):
        mask = (1 << (end - start + 1)) - 1
        return (arr >> start) & mask
    
    # Calculate cloud cover from scene classification data.
    cloud = extract_bits(qa, 1, 4) > 0
    cloud_prop = cloud.mean() * 100
    if cloud_prop > cloud_cover:
        return launch_data
    
    # Avoid scenes which has more than 20% invalid pixels.
    invalid = extract_bits(qa, 0, 0) > 0
    invalid_prop = invalid.mean() * 100
    if invalid_prop > 20:
        return launch_data
    
    data_dict_out = launch_data_dict.copy()
    data_dict_out["cloud_prop"] = cloud_prop

    if scene == "tar":
        delta_t_secs = (overpass_dt - launch_data_dict["launch_time"]).total_seconds()
        data_dict_out["delta_t_secs"] = delta_t_secs
        print(f'Site = {site}, Rocket = {launch_data_dict["launch_rocket"]} on {launch_data_dict["launch_time"]}. Cloud-cover = {cloud_prop:.2f}%, ΔT = {(delta_t_secs/60):.1f} mins. Platform = {satellite}.')
    else:
        data_dict_out["overpass_time"] = overpass_dt
        print(f'{overpass_dt}. Cloud-cover = {cloud_prop:.2f}%. Platform = {satellite}.')
    launch_data.append(data_dict_out)

    return launch_data