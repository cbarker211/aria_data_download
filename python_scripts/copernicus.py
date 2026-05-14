from sentinelhub.data_collections import DataCollection
from sentinelhub.config import SHConfig
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType
from sentinelhub.geo_utils import bbox_to_dimensions
from sentinelhub.api.process import SentinelHubRequest

# Regular imports
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import sys

def get_scl(bbox, config, time_start, time_end):

    path = Path(__file__).parent.parent / "evalscripts" / f"evalscript_scl.js"
    evalscript_scl = path.read_text()

    request_scl_data = SentinelHubRequest(
        data_folder='temp_dir', evalscript=evalscript_scl, bbox=bbox, size=bbox_to_dimensions(bbox, resolution=60), config=config,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                time_interval=(time_start, time_end),
                other_args={"id":"S2L2A"},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    )
    scldata = request_scl_data.get_data()[0]

    return scldata

def search_copernicus(data_dict_list, data_dict, bbox, scene, cloud_cover, site):
    """ Searches the Copernicus Sentinel-2 catalog for overpasses matching the time window and location of interest, and filters them by cloud cover.
        Returns a list of dictionaries containing information about the overpasses that match the criteria.
        Ran for both target and reference scenes."""

    # Load configuration file
    config = SHConfig()
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    catalog = SentinelHubCatalog(config=config)

    # Error check to make sure credientials have been provided:
    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
    
    # First search over the time window. Set a coarse cloud cover filter to speed things up.
        
    launch_time = data_dict["launch_time"]
    time_start =  (launch_time + timedelta(hours=0)).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end   =  (launch_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    search_iterator = catalog.search(DataCollection.SENTINEL2_L1C, bbox=bbox, time = (time_start, time_end), filter="eo:cloud_cover<30")
    results = list(search_iterator)
    if not results:
        return data_dict_list

    # Isolate the best overpass.
    # Each tile is part of the complete datatake, split into tiles.
    # Each tile is 110x110km2 on a 100x100km2 grid, so with a 10km overlap.
    # Therefore sometimes the site of interest is in the overlap, but its essentially the same data.
    # So we only want to keep multiple overpasses if they are from different platforms. 
    # Keep only the earliest result per platform (i.e. one per overpass).

    earliest_per_overpass= {}
    for idx, result in enumerate(results):

        # Get time and round to nearest minute
        dt = datetime.fromisoformat(result["properties"]["datetime"].replace("Z", "+00:00"))
        dt_min = (dt + timedelta(seconds=30)).replace(second=0, microsecond=0)

        # Add to dict only if it's the earliest for that platform and minute.
        key = (result["properties"]["platform"], dt_min)

        if key not in earliest_per_overpass or dt < earliest_per_overpass[key][1]:
            earliest_per_overpass[key] = (idx, dt)

    results = [results[idx] for idx, _ in earliest_per_overpass.values()]

    iterator = tqdm(results) if len(results) > 1 else results
    # Loop over the results to find the overpasses with cloud cover under a threshold.
    for result in iterator:
        
        # Copy so that we can change data_dict each time.
        data_dict_out = data_dict.copy()

        # Set the time window.
        overpass_dt = datetime.fromisoformat(result['properties']['datetime'])
        op_time_start = (overpass_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        op_time_end   = (overpass_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
        # https://sentiwiki.copernicus.eu/web/s2-processing
        # The classification map is produced for each Sentinel-2 Level-2A product at 20 m and resamples to 60 m resolution.
        # This is searching the level 2 (processed) data, which has the scene classification layer needed for cloud filtering.
        # Request at 20m resolution to check cloud cover.

        scldata = get_scl(bbox, config, op_time_start, op_time_end)
        if scldata.size == 0:
            raise ImportError("No data found.")
        
        # Calculate cloud cover from scene classification data.
        scl = scldata[:, :]
        cld_mask = np.isin(scl, [3, 8, 9, 10])
        cloud_prop = cld_mask.mean() * 100
        if cloud_prop > cloud_cover:
            continue
        
        # Avoid scenes which has more than 20% invalid pixels.
        invalid_mask = np.isin(scl, [0, 1])  
        invalid_prop = (invalid_mask).mean() * 100   
        if invalid_prop > 20:
            if scene == "tar":
                print(f"Scene on {data_dict["launch_time"]} has {invalid_prop:.2f}% invalid pixels. Skipping.")
            if scene == "ref":
                print(f"Scene on {overpass_dt} has {invalid_prop:.2f}% invalid pixels. Skipping.")
            continue

        data_dict_out["cloud_prop"] = cloud_prop

        if scene == "tar":
            delta_t_secs = (overpass_dt - data_dict["launch_time"]).total_seconds()
            data_dict_out["delta_t_secs"] = delta_t_secs
            print(f'Site = {site}, Rocket = {data_dict["launch_rocket"]} on {data_dict["launch_time"]}. Cloud-cover = {cloud_prop:.2f}%, ΔT = {(delta_t_secs/60):.1f} mins. Platform = {result["properties"]["platform"]}.')
        else:
            data_dict_out["overpass_time"] = overpass_dt
            print(f'{overpass_dt}. Cloud-cover = {cloud_prop:.2f}%. Platform = {result["properties"]["platform"]}.')
        data_dict_list.append(data_dict_out)
            
    return data_dict_list

def download_copernicus(data_out, res, launch_time, time_start, time_end, ind, lon_range, lat_range, bbox, site, scene):

    # Load configuration file
    config = SHConfig()
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

    # Error check to make sure credentials have been provided:
    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

    evalscript = Path(f"evalscripts/evalscript.js").read_text()
    request_res_dep_data = SentinelHubRequest(
        data_folder='download', evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                time_interval=(time_start, time_end),
                other_args={"id":"S2L1C"},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=bbox_to_dimensions(bbox, resolution=res), config=config,
    )
    # Get and check data:
    requested_data = request_res_dep_data.get_data()
    data_out[ind,:,:,:-1] = requested_data[0]
    if len(requested_data) > 1:
        raise ValueError('More than one image returned. Please check the data and code.')
    elif len(requested_data) == None:
        raise ImportError("No data found.")

    # Get the scl data 
    scldata = get_scl(bbox, config, time_start, time_end)
    if scldata.size == 0:
        raise ImportError("No data found.")
    data_out[ind,:,:,-1] = scldata
    cld_mask = np.isin(scldata, [3, 8, 9, 10])
    
    # Save an image of the launch site overpass for a visual check.
    print("Printing image.")
    fig, ax = plt.subplots(figsize=(15, 10))
    extent = (lon_range[0], lon_range[1], lat_range[0], lat_range[1])       
    rgb = np.stack([
        requested_data[0][:, :, 8],  # Red
        requested_data[0][:, :, 7],  # Green
        requested_data[0][:, :, 6],  # Blue
    ], axis=-1)
    rgb = np.clip(rgb / 5000.0, 0, 1)
    
    ax.imshow(rgb, extent=extent)

    # Create red overlay (RGBA)
    overlay = np.zeros((*cld_mask.shape, 4))  # 4 = RGBA
    overlay[..., 0] = 1.0          # Red channel
    overlay[..., 3] = cld_mask * 0.4  # Alpha (transparency)

    # Plot overlay
    ax.imshow(overlay, extent=extent)

    path = f"site_images/{site.replace('/', '')}/{scene}"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{launch_time.strftime("%Y-%m-%d")}_{res}.png")
    plt.close()

    return data_out