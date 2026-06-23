from datetime import datetime, timedelta 
from planet import data_filter
from tqdm import tqdm
import sys

def search_planet(launch_data, launch_data_dict, planet, geom, scene, cloud_cover, site, platform):

    if scene == "tar":
        launch_time = launch_data_dict["launch_time"]
        time_start =  launch_time
        time_end   =  (launch_time + timedelta(hours=1))
    elif scene == "ref":
        time_start = launch_data_dict["time_start"]
        time_end   = launch_data_dict["time_end"]
    
    if platform == "sd":
    
        sfilter = data_filter.and_filter([data_filter.geometry_filter(geom),
                                        data_filter.date_range_filter('acquired',gt=time_start,lt=time_end),
                                        data_filter.string_in_filter('instrument', ['PSB.SD']),
                                        data_filter.range_filter('cloud_cover', lte=cloud_cover/100),
                                    ])

        items = list(planet.data.search(['PSScene'], search_filter=sfilter, limit=50))


    elif platform == "tn":
        sfilter = data_filter.and_filter([data_filter.geometry_filter(geom),
                                        data_filter.date_range_filter('acquired',gt=time_start,lt=time_end),
                                    ])

        items = list(planet.data.search(['TanagerScene'], search_filter=sfilter, limit=50))
    
    # Return if no images are found
    n_images = len(items)
    if n_images == 0:
        return launch_data

    times = sorted([datetime.fromisoformat(item["properties"]["acquired"].replace("Z", "+00:00")) for item in items])
    # TODO: Need to filter for clouds and invalid pixels.

    collapsed = []
    last_time = None
    for i in range(len(times)):
        if last_time is None or (times[i] - last_time) > timedelta(seconds=60):  # >60 s = distinct pass
            collapsed.append(i)
            last_time = times[i]
        valid_indices = collapsed

    if len(valid_indices) > 1:
        iterator = tqdm(valid_indices)
    else:
        iterator = valid_indices

    for i in iterator:

        # Copy so that we can change data_dict each time.
        if scene == "tar":
            data_dict_out = launch_data_dict.copy()
        elif scene == "ref":
            data_dict_out = {}

        #data_dict_out["cloud_prop"] = cloud_props[i]

        # Get metadata.
        overpass_dt = times[i]
        
        if scene == "tar":
            delta_t_secs = (overpass_dt - launch_data_dict["launch_time"]).total_seconds()
            data_dict_out["delta_t_secs"] = delta_t_secs
            print(f'Site = {site}, Rocket = {launch_data_dict["launch_rocket"]} on {launch_data_dict["launch_time"]}, ΔT = {(delta_t_secs/60):.1f} mins.')#. Cloud-cover = {cloud_props[i]:.2f}%')
        else:
            data_dict_out["overpass_time"] = overpass_dt
            print(f'{overpass_dt})')#. Cloud-cover = {cloud_props[i]:.2f}%.')
        launch_data.append(data_dict_out)
    
    return launch_data