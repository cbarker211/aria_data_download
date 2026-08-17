import ee
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import sys
import requests
from io import BytesIO
import os
import xarray as xr
from tqdm import tqdm
from omnicloudmask import predict_from_array
import time

def search_gee(launch_data, launch_data_dict, landsat, bbox, scene, threshold, site):

    region = ee.Geometry.Rectangle(bbox)

    # Set up the time range.
    if scene == "tar":
        launch_time = launch_data_dict["launch_time"]
        time_start =  (launch_time).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_end   =  (launch_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif scene == "ref":
        time_start = launch_data_dict["time_start"].strftime("%Y-%m-%dT%H:%M:%SZ")
        time_end   = launch_data_dict["time_end"].strftime("%Y-%m-%dT%H:%M:%SZ")

    # Search Landsat 8/9 Collection 2 Tier 1 TOA Reflectance
    landsat_filtered = landsat.filterBounds(region) \
                           .filterDate(time_start, time_end)
    
    # Return if no images are found
    n_images = landsat_filtered.size().getInfo()
    if n_images == 0:
        return launch_data, 0, 0, None

    # Calculate metadata over all images.
    # This saves having to do it for each image separately in a loop.
    def add_cloud_metrics(image):

        qa = image.select('QA_PIXEL')
        mask = (ee.Number(1).leftShift(1)   # dilated cloud
           .add(ee.Number(1).leftShift(2))  # cirrus
           .add(ee.Number(1).leftShift(3))  # cloud
           .add(ee.Number(1).leftShift(4))  # cloud shadow
        )
        cloud = qa.bitwiseAnd(mask).gt(0).rename('cloud')

        snow = qa.bitwiseAnd(ee.Number(1).leftShift(5)).gt(0).rename('snow')
        stats = ee.Dictionary({'cloud': 1, 'snow': 1}).combine(
            cloud.addBands(snow).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                maxPixels=1e8
            ),
            overwrite=True
        )

        # This divides the valid pixels in the region by the total pixels.
        # Then subtract from one and multiply by 100 to get % invalid pixels.
        invalid_frac = ee.Number(1).subtract(
            image.geometry().intersection(region).area()
                .divide(region.area())
        ).multiply(100)

        return image.set({
            'invalid_prop': invalid_frac,
            'cloud_prop':   ee.Number(stats.get('cloud')).multiply(100),
            'snow_prop':    ee.Number(stats.get('snow')).multiply(100),
        })

    landsat_filtered = (landsat_filtered.map(add_cloud_metrics).sort('system:time_start'))

    times               = landsat_filtered.aggregate_array('system:time_start').getInfo() 
    satellites          = landsat_filtered.aggregate_array('SPACECRAFT_ID').getInfo()
    cloud_props         = landsat_filtered.aggregate_array('cloud_prop').getInfo()
    snow_props          = landsat_filtered.aggregate_array('snow_prop').getInfo()
    invalid_props       = landsat_filtered.aggregate_array('invalid_prop').getInfo() 

    # Work out the shape of the image.
    test_img  = landsat_filtered.first().select("QA_PIXEL")
    qa_sample = test_img.sampleRectangle(region=region, defaultValue=65535).getInfo()
    qa_array = np.array(qa_sample["properties"]["QA_PIXEL"])
    shape = qa_array.shape  # (height, width)

    if scene == "tar":
        proj = landsat_filtered.first().select("B1").projection().getInfo()
    else:
        proj = None

    # Filter out cloudy scenes or ones with lots of invalid pixels.
    valid_indices = [i for i in range(n_images) if cloud_props[i] <= threshold and invalid_props[i] <= 40 and snow_props[i] <= threshold]

    collapsed = []
    last_time = None
    for i in valid_indices:
        if last_time is None or (times[i] - last_time) > 60_000:  # >60 s = distinct pass
            collapsed.append(i)
            last_time = times[i]
    valid_indices = collapsed

    if len(valid_indices) > 1:
        iterator = tqdm(valid_indices)
    else:
        iterator = valid_indices #range(n_images)
    
    for i in iterator:

        # Copy so that we can change data_dict each time.
        if scene == "tar":
            data_dict_out = launch_data_dict.copy()
        elif scene == "ref":
            data_dict_out = {}

        data_dict_out["cloud_prop"] = cloud_props[i]

        # Get metadata.
        time_ms = times[i]
        satellite = satellites[i]
        overpass_dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
        
        if scene == "tar":
            delta_t_secs = (overpass_dt - launch_data_dict["launch_time"]).total_seconds()
            data_dict_out["delta_t_secs"] = delta_t_secs
            print(f'Site = {site}, Rocket = {launch_data_dict["launch_rocket"]} on {launch_data_dict["launch_time"]}. Cloud-cover = {cloud_props[i]:.2f}%, ΔT = {(delta_t_secs/60):.1f} mins. Platform = {satellite}.')
        else:
            data_dict_out["overpass_time"] = overpass_dt
            print(f'{overpass_dt}. Cloud-cover = {cloud_props[i]:.2f}%. Platform = {satellite}.')
        launch_data.append(data_dict_out)
    
    return launch_data, shape[0], shape[1], proj

def download_gee(landsat, launch_time, time_start, time_end, bbox, site, scene, proj):

    print(launch_time)
    region = ee.Geometry.Rectangle(bbox)

    # Search Landsat 8/9 Collection 2 Level-2 (Surface Reflectance)
    landsat_filtered = landsat.filterBounds(region).filterDate(time_start, time_end)  
    
    count = landsat_filtered.size().getInfo()
    if count == 0:
        raise ImportError("No data found.")
    elif count > 1:
        times = landsat_filtered.aggregate_array('system:time_start').getInfo()  # ms epoch
        stamps = [datetime.fromtimestamp(t / 1000, tz=timezone.utc)
                        .strftime('%Y-%m-%d %H:%M') for t in times]
        print(f'Warning: More than one image returned ({count}). '
            f'Overpass times: {", ".join(stamps)}. Taking the first image.')
    
    landsat_image = landsat_filtered.first() # Isolate the image.
    time_ms = landsat_image.get('system:time_start').getInfo() # Get metadata.
    overpass_dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
    delta_t_secs = (overpass_dt - launch_time).total_seconds()

    # Select and download specific variables.
    bands = ["SAA","SZA","VAA","VZA","B1","B2","B3","B4","B5","B6","B7","B8","B10","B11","QA_PIXEL"]  
    landsat_image = landsat_image.select(bands)
    url = landsat_image.getDownloadURL({
        'region': region,     # specified region only
        'crs': proj["crs"],
        'crs_transform': [int(v) for v in proj["transform"]],
        "format": "NPY"       # numpy
    })

    def _download_npy(url, retries=6):
        for attempt in range(retries):
            response = requests.get(url)
            if response.status_code == 200 and response.content[:6] == b'\x93NUMPY':
                return np.load(BytesIO(response.content))
            print(f"  attempt {attempt+1}: status={response.status_code}, {response.content[:120]!r}")
            time.sleep(2 ** attempt)
        raise RuntimeError(f"GEE download failed after {retries} attempts: {url}")

    data = _download_npy(url)
    ds = xr.Dataset({band: (("y", "x"), data[band]) for band in data.dtype.names})
    data_out = ds.to_array(dim="band").transpose("y", "x", "band").values
    for i in range(4): # Scale the angles to degrees.
        data_out[..., i] = data_out[..., i] / 100

    # Save an image of the launch site overpass for a visual check.
    fig, ax = plt.subplots(figsize=(12, 10))

    def stackrgb(ds, bands, axis=-1):
        return np.stack([ds[band].values for band in bands], axis = axis)

    rgb = stackrgb(ds, ["B4","B3","B2"], -1)
    rgb = np.clip(rgb / 0.3, 0, 1) # Clip so we can actually see the image.
    ax.imshow(rgb)

    # OmniCloudMask mask (blue)
    #rgbnir = stackrgb(ds, ["B4","B3","B5"], 0)
    #mask = predict_from_array(rgbnir)[0]   # (H, W)
    #mask = mask.astype(bool)
    #omni_rgba = np.zeros((*mask.shape, 4))
    #omni_rgba[mask] = [0, 0, 1, 0.2]
    #ax.imshow(omni_rgba)
    #print(mask.mean() * 100)

    # Overlay snow mask as semi-transparent green
    qa = ds["QA_PIXEL"].values
    mask = (1 << 5)
    snow = (qa & mask) > 0
    snow_rgba = np.zeros((*snow.shape, 4))
    snow_rgba[snow] = [0, 1, 0, 0.2]
    ax.imshow(snow_rgba)
    snow_pct = snow.mean() * 100

    # Overlay cloud mask as semi-transparent red
    mask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    cloud = (qa & mask) > 0
    cloud_rgba = np.zeros((*cloud.shape, 4))
    cloud_rgba[cloud] = [1, 0, 0, 0.2]
    ax.imshow(cloud_rgba)
    cloud_pct = cloud.mean() * 100

    path = f"site_images/landsat/{site.replace('/', '')}/{scene}"
    if scene == "tar":
        title = f'Landsat 8/9 Scene: {scene}, Time after launch: {(delta_t_secs/60):.1f} mins, Cloud Cover: {cloud_pct:.2f}%'
    elif scene == "ref":
        datestr = overpass_dt.strftime("%m/%d/%Y, %H:%M:%S")
        title = f'Landsat 8/9 Scene: {scene}, Date: {datestr}, Cloud Cover: {cloud_pct:.2f}%, Snow Cover: {snow_pct:.2f}%'
    if not os.path.exists(path):
        os.makedirs(path)
    
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"{path}/{site}_{launch_time.strftime("%Y-%m-%d")}_{scene}_landsat.png")
    plt.close()
    
    return data_out