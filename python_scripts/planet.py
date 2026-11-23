from datetime import datetime, timedelta 
from planet import data_filter
from tqdm import tqdm
from shapely.geometry import shape
import numpy as np
import sys

def search_planet(launch_data_dict, planet, geom, scene, cloud_cover, site):
    
    launch_data = []
    
    # Set up the time filters.
    if scene == "tar":
        time_filter = data_filter.or_filter([data_filter.date_range_filter('acquired', gt=l["launch_time"], lt=l["launch_time"] + timedelta(hours=1)) for l in launch_data_dict])
    elif scene == "ref":
        time_filter = data_filter.date_range_filter('acquired', gt=launch_data_dict["time_start"], lt=launch_data_dict["time_end"])
    else:
        return launch_data
    
    # Search for launch0-coincident overpasses.
    aoi = shape(geom)
    sfilter = data_filter.and_filter([data_filter.geometry_filter({"type": "Point", "coordinates": [aoi.centroid.x, aoi.centroid.y]}),
                                        time_filter,
                                        data_filter.string_in_filter('instrument', ['PSB.SD']),
                                        data_filter.range_filter('clear_percent', gte=(100-cloud_cover)),
                                        data_filter.range_filter('anomalous_pixels', lte=0.4),
                                        data_filter.string_in_filter('ground_control', ['true']),
                                        data_filter.string_in_filter('quality_category', ['standard']),
                                ])
    items = list(planet.data.search(['PSScene'], search_filter=sfilter, limit=0, sort="acquired asc"))
    
    # Return if no images are found
    if len(items) == 0:
        return launch_data
        
    clear_items = []
    last_time = None
    for item in items:
        
        # Match scene to launch
        acq_time = datetime.fromisoformat(item["properties"]["acquired"].replace("Z", "+00:00"))
        launch = None
        if scene == "tar":
            launch = next((l for l in launch_data_dict if timedelta(0) < acq_time - l["launch_time"] < timedelta(hours=1)), None)
            if launch is None:
                print("No launch found")
                continue
        
        # Coverage Check
        covered = shape(item["geometry"]).intersection(aoi).area / aoi.area
        if covered < 0.5:   # i.e. more than 50% of the bbox has no data
            print(f"No coverage ({covered:.2f}) {acq_time}")
            continue
        
        # Time Check
        if last_time is None or (acq_time - last_time) > timedelta(seconds=60):  # >60 s = distinct pass
            clear_items.append((item, launch))
            last_time = acq_time
                
    if len(clear_items) == 0:
        return launch_data
    elif len(clear_items) > 1:
        iterator = tqdm(clear_items)
    else:
        iterator = clear_items

    for item, launch in iterator:

        # Copy so that we can change data_dict each time.
        if scene == "tar":
            data_dict_out = launch.copy()
        elif scene == "ref":
            data_dict_out = {}
        else:
            return launch_data

        # Get metadata.
        overpass_dt = datetime.fromisoformat(item["properties"]["acquired"].replace("Z", "+00:00"))
        
        if scene == "tar":
            delta_t_secs = (overpass_dt - launch["launch_time"]).total_seconds()
            data_dict_out["delta_t_secs"] = delta_t_secs
            print(f'Site = {site}, Rocket = {launch["launch_rocket"]} on {launch["launch_time"]}, ΔT = {(delta_t_secs/60):.1f} mins.')
        else:
            data_dict_out["overpass_time"] = overpass_dt
            print(f'{overpass_dt})')
        data_dict_out["item_id"] = item["id"]
        launch_data.append(data_dict_out)
    
    return launch_data