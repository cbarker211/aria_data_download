import sys
import numpy as np
from sentinelhub.config import SHConfig
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType
from sentinelhub.geo_utils import bbox_to_dimensions
from sentinelhub.api.process import SentinelHubRequest
import json
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
import xarray as xr
from tqdm import tqdm
from utils import build_grid
import pandas as pd

# Units:
# DN = 10000 * REFLECTANCE
# AOT = DN / 1000

# Maybe also download band B04 (665 nm) where NO2 absorption small
# (About 30-times less than at B01 and B02)

# Table showing data available: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/
# https://sentiwiki.copernicus.eu/web/s2-products

# Scene classification data:
# 0=no data
# 1=saturated/defective pixel
# 2=topographic shadows
# 3=cloud shadow
# 4=vegetation
# 5=not vegetated
# 6=water
# 7=unclassified
# 8=cloud with medium probability
# 9=cloud with high prob
# 10=thin cirrus
# 11=snow/ice

# Data mask:
# 0=nodata
# 1=has data
class find_data:

    def __init__(self):
        """This sets up the class"""

        # Target resolutions of output
        # (60 m for B01 or UB and 10 m for B02 or blue band):
        # B01: ultrablue (442.7,442.3 nm) (S2A, S2B)
        # B02: blue (492.4 nm, 492.1 nm)
        self.target_res = [10, 60]

        ##########################
        # Set up Sentinel Hub API
        ##########################

        # Load configuration file
        self.config = SHConfig()
        self.config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
        self.config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        self.catalog = SentinelHubCatalog(config=self.config)

        # Error check to make sure credientials have been provided:
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

        #####################
        # Define evalscripts 
        #####################

        def load_evalscript(name, band=None):
            path = Path(__file__).parent / "evalscripts" / f"{name}.js"
            script = path.read_text()
            if band:
                script = script.replace("BAND_PLACEHOLDER", band)
            return script

        self.evalscript = {res: "" for res in self.target_res}
        self.evalscript[60] = load_evalscript("evalscript_res", band="B01")
        self.evalscript[10] = load_evalscript("evalscript_res", band="B02")
        self.evalscript_scl   = load_evalscript("evalscript_scl")

    def find_coincidences(self, bbox, deltat_minus, deltat_plus, launch_time, launch_data, site, cloud_cover):
        """ This functions checks if there's a Sentinel-2 overpass within a certain time period
            and if so, checks the cloud cover from the scene classification layer. 
            If the cloud cover is less than 20%, it returns the launch time, cloud cover, and time difference between overpass and launch, 
            as well as the scene classification data for that overpass."""

        # First query if there's data on this day to minimise queries.
        # This is searching the level 1 (raw) data.
        time_start =  (launch_time + timedelta(hours=deltat_minus)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_end   =  (launch_time + timedelta(hours=deltat_plus)).strftime("%Y-%m-%dT%H:%M:%SZ")

        search_iterator = self.catalog.search(DataCollection.SENTINEL2_L1C, bbox=bbox, time = (time_start, time_end))
        results = list(search_iterator)
        if not results:
            return launch_data 
        
        # Each tile is part of the complete datatake, split into tiles.
        # Each tile is 110x110km2 on a 100x100km2 grid, so with a 10km overlap.
        # Therefore sometimes the site of interest is in the overlap, but its essentially the same data.
        # So we only want to keep multiple overpasses if they are from different platforms. 
        # Keep only the earliest result per platform (i.e. one per overpass).
        platforms = list(result["properties"]["platform"] for result in results)
        times     = list(result["properties"]["datetime"] for result in results)
        unique_platforms = np.unique(platforms)

        earliest_per_platform = {}
        for idx, (platform, time) in enumerate(zip(platforms, times)):
            if platform not in earliest_per_platform or time < earliest_per_platform[platform][1]:
                earliest_per_platform[platform] = (idx, time)
        results_indexed = [results[idx] for idx, _ in earliest_per_platform.values()]

        if len(results_indexed) > 1:
            print(f"{len(results_indexed)} overpasses found from {unique_platforms}.")

        # Set up the dictionary to contain all the information about the overpasses found.
        launch_data_dict = {"launch_time": launch_time,
                            "cloud_prop": [],
                            "delta_t_secs": []
                            }
        
        #for res in self.target_res:
        #    launch_data_dict[str(res)] = []

        # Loop over each overpass and check for cloud cover.
        for result in results_indexed:

            overpass_dt = datetime.fromisoformat(result['properties']['datetime'])
            delta_t_secs = (overpass_dt - launch_time).total_seconds()
            op_time_start = (overpass_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            op_time_end   = (overpass_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
            # https://sentiwiki.copernicus.eu/web/s2-processing
            # The classification map is produced for each Sentinel-2 Level-2A product at 20 m and resamples to 60 m resolution.
            # This is searching the level 2 (processed) data, which has the scene classification layer needed for cloud filtering.
            # Request at 20m resolution to check cloud cover, then resample at 60m and 10m.
            request_scl_data = SentinelHubRequest(
                data_folder='temp_dir', evalscript=self.evalscript_scl, bbox=bbox, size=bbox_to_dimensions(bbox, resolution=20), config=self.config,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=self.config.sh_base_url),
                        time_interval=(op_time_start, op_time_end),
                        other_args={"id":"S2L2A"},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            )
            
            scl20 = request_scl_data.get_data()[0]
            if scl20.size == 0:
                continue
            
            # Calculate cloud cover from scene classification data.
            cld_mask = np.isin(scl20, [3, 8, 9, 10])
            cloud_prop = cld_mask.mean() * 100
            if cloud_prop > cloud_cover:
                continue
                
            print(f'Site = {site} on {launch_time}. Cloud-cover = {cloud_prop:.2f}%, ΔT = {(delta_t_secs/60):.1f} mins.')

            launch_data_dict["cloud_prop"].append(cloud_prop)
            launch_data_dict["delta_t_secs"].append(delta_t_secs)

            # Resample the scl data to the target resolutions.
            #da = xr.DataArray(scl20, dims=("y","x"))
            #scl10 = da.isel(y=np.repeat(np.arange(da.sizes["y"]), 2),
            #                x=np.repeat(np.arange(da.sizes["x"]), 2))  
            #scl60 = (da
            #        .coarsen(x=3, y=3, boundary="trim")
            #        .reduce(lambda x, axis: stats.mode(x, axis=axis, keepdims=False).mode)
            #    )
            #launch_data_dict["10"].append(scl10) 
            #launch_data_dict["60"].append(scl60) 

        if len(launch_data_dict["cloud_prop"]) > 0:
            launch_data.append(launch_data_dict)
                    
        return launch_data
    
    def retrieve_data(self, bbox, deltat_minus, deltat_plus, launch_data, site, loc_name, lat_range, lon_range, scene):

        # Launch data is now a list, where each item is a launch that had an overpass (regardless of whether it had clouds).
        # The length of the lists within each launch will tell you how many cloud-free overpasses were found.

        # First go through and remove launches where no cloud-free overpasses were found.
        # Then build an array of dimensions (launch_time, overpass_number, lat, lon, variable)

        # Setup an output array to hold the data to save. 
        max_overpasses = max(len(l["delta_t_secs"]) for l in launch_data)

        sizes = {res: bbox_to_dimensions(bbox, resolution=res) for res in self.target_res}
        data_out = {res: np.full((len(launch_data), max_overpasses, sizes[res][1], sizes[res][0], 5), np.nan, dtype=np.float32) for res in self.target_res}

        for res in self.target_res:

            for launch_ind, launch in enumerate(launch_data):

                if len(launch["cloud_prop"]) == 0:
                    continue

                launch_time = launch["launch_time"]

                for overpass in range(len(launch["delta_t_secs"])):

                    if deltat_minus is None:
                        t_minus = launch["delta_t_secs"][overpass] / 3600 - 1
                        t_plus  = launch["delta_t_secs"][overpass] / 3600 + 1
                    else:
                        t_minus = deltat_minus
                        t_plus  = deltat_plus

                    time_start = (launch_time+timedelta(hours=t_minus)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    time_end   = (launch_time+timedelta(hours=t_plus)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
                    request_res_dep_data = SentinelHubRequest(
                        data_folder='download', evalscript=self.evalscript[res],
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=self.config.sh_base_url),
                                time_interval=(time_start, time_end),
                                other_args={"id":"S2L1C"},
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        bbox=bbox, size=bbox_to_dimensions(bbox, resolution=res), config=self.config,
                    )
                    # Get and check data:
                    requested_data = request_res_dep_data.get_data()
                    data_out[res][launch_ind,overpass,:,:,:] = requested_data[0]
                    if len(requested_data) > 1:
                        raise ValueError('More than one image returned. Please check the data and code.')
                    elif len(requested_data) == None:
                        raise ImportError("No data found.")
                    
                    # Save an image of the launch site overpass for a visual check.
                    fig, ax = plt.subplots(2,figsize=(15, 10),subplot_kw={"projection": ccrs.PlateCarree()})
                    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
                    data = [requested_data[0][:,:,3], requested_data[0][:,:,4]]
                    for i in range(2):
                        ax[i].imshow(data[i], extent=extent, origin="upper",transform=ccrs.PlateCarree())
                    
                    plt.savefig(f'site_images/{site.replace("/","")}_{launch_time.strftime("%Y-%m-%d")}_{res}_{scene}_{overpass}.png')
                    plt.close()
            
            # Save data for this site
            self.save_to_netcdf(site, res, sizes, data_out, launch_data, lat_range, lon_range, loc_name, scene)
            print(f'{site}-{res} written to disk')

    def data_search(self):
        """This function downloads the Sentinel-2 data for the launch sites and times specified in the sites.json file."""

        # Open the sites file.
        site_data = json.load(open("sites.json"))
        sites = list(site_data.keys()) 
        sites = ["CC_LC40"]
        df_sites = pd.read_csv('coincidences.csv', index_col='Site')
        sites_data = df_sites.to_dict(orient='index')

        # Loop over each site and each launch at that site, and look for Sentinel-2 data on that day.
        # Starting off with the first quarter of launch sites.
        for site in sites:

            # Get the full spaceport name and list of launch times.
            loc_name = site_data[site]['name']
            launch_times = site_data[site]['launch_times']
            print(f"Searching at {loc_name}.")

            # Set up grid box around site:
            bbox, lat_range, lon_range = build_grid(site_data[site]['lat'], site_data[site]['lon'])   

            ###################################################################################
            # First check for which times have an overpass within 1 hour with low cloud cover.
            ###################################################################################
            
            launch_data = []

            for time in tqdm(launch_times, total = len(launch_times), desc="Processing rows"):

                launch_time = datetime.fromisoformat(time).replace(tzinfo=timezone.utc)
                launch_data = self.find_coincidences(bbox, 0, 1, launch_time, launch_data, site, 20)

            sites_data[site]["Targets"] = len(launch_data)
            if len(launch_data) == 0:
                continue
        
            self.retrieve_data(bbox, 0, 1, launch_data, site, loc_name, lat_range, lon_range, "target")

            ############################################
            # Now look for overpasses before the launch. 
            ############################################

            # Only look either side of the launches we found.
            times_to_check = [t["launch_time"] for t in launch_data]
            launch_data = []
            # The revisit time is latitude-dependent, so choose maximum 5 days either side.
            # We only find the nearest overpass anyway.
            for time in times_to_check:

                launch_data_for_time = []
                
                # First check 5 days either side with cloud cover at 20%
                launch_data_for_time  = self.find_coincidences(bbox, -120, -5, time, launch_data_for_time, site, 20)
                if len(launch_data_for_time ) == 0:
                    launch_data_for_time  = self.find_coincidences(bbox, 5, 120, time, launch_data_for_time, site, 20)
                    
                if len(launch_data_for_time ) > 0:
                    launch_data.extend(launch_data_for_time)
                    sites_data[site]["References"] = len(launch_data_for_time )
                    continue

                # Then repeatedly widen the time window at 50% cloud cover until a match is found
                window_start = 5
                window_size = 115
                while len(launch_data_for_time) == 0:
                    window_end = window_start + window_size
                    launch_data_for_time = self.find_coincidences(bbox, -window_end, -window_start, time, launch_data_for_time, site, 50)
                    if len(launch_data_for_time) == 0:
                        launch_data_for_time = self.find_coincidences(bbox, window_start, window_end, time, launch_data_for_time, site, 50)
                    window_start = window_end
                    window_size = 120
                
                launch_data.extend(launch_data_for_time)
                sites_data[site]["References"] = len(launch_data)
            
            self.retrieve_data(bbox, None, None, launch_data, site, loc_name, lat_range, lon_range, "reference")

        df = pd.DataFrame.from_dict(sites_data, orient='index')
        df.index.name = 'Site'
        df.to_csv('coincidences.csv')

    def save_to_netcdf(self, site, res, sizes, data_out, launch_data, lat_range, lon_range, loc_name, scene):

        filename = f'sentinel2-l1c-band-l2a-slc-{site.replace("/","")}-{res}-{scene}.nc'
        
        # Set up strings for saving to netCDF
        variables = {10: ["sza", "vza", "dataMask", "B02", "B04"],
                    60: ["sza", "vza", "dataMask", "B01", "B04"]}
        units = ["degrees", "degrees", "unitless", "DN", "DN"]
        long_name_strs = ['Sentinel-2-l1c solar zenith angle', 
                    'Sentinel-2-l1c viewing zenith angle',
                    'Sentinel-2-l1c data mask (0=nodata, 1=data)',
                    '',
                    'Sentinel-2-l1c B04 TOA reflectance in DN (DN=1e4*reflectance)']
        long_names = {10: long_name_strs,
                        60: long_name_strs}
        long_names[10][3] = 'Sentinel-2-l1c B02 TOA reflectance in DN (DN=1e4*reflectance)'
        long_names[60][3] = 'Sentinel-2-l1c B01 TOA reflectance in DN (DN=1e4*reflectance)'

        y_coords  = np.arange(sizes[res][0], dtype=np.int32)
        x_coords  = np.arange(sizes[res][1], dtype=np.int32)

        # Work out the maximum number of overpasses for all the data searched.
        max_overpasses = max(len(l["delta_t_secs"]) for l in launch_data)
        # Create arrays for all overpasses.
        delta_t_secs_padded = np.full((len(launch_data), max_overpasses), np.nan, dtype=np.float32)
        cloud_prop_padded   = np.full((len(launch_data), max_overpasses), np.nan, dtype=np.float32)
        #scl_data_padded     = np.full((len(launch_data), sizes[res][0], sizes[res][1], max_overpasses), np.nan, dtype=np.float32)

        for ind, l in enumerate(launch_data):
            no_overpasses = len(l["delta_t_secs"])
            delta_t_secs_padded[ind,:no_overpasses]     = l["delta_t_secs"]
            cloud_prop_padded  [ind,:no_overpasses]     = l["cloud_prop"]  
            #scl_data_padded    [ind,:,:,:no_overpasses] = l[str(res)]        

        data_vars = {}
        for ind, var in enumerate(variables[res]):
            data_vars[var] = xr.Variable(
                dims=("time", "overpass", "lat", "lon"),
                data=data_out[res][:,:,:,:,ind],
                attrs={"long_name": long_names[res][ind], "units": units[ind]}
            )

        data_vars["cloud_prop"] = xr.Variable(
            dims = ("time", "overpass"),
            data = cloud_prop_padded,
            attrs= {"long_name": "Cloud cover proportion in bbox", "units": "%"}
        )

        data_vars["delta_t_secs"] = xr.Variable(
            dims = ("time", "overpass"),
            data = delta_t_secs_padded,
            attrs= {"long_name": "Time difference between overpass and launch", "units": "seconds"}
        )

        #data_vars["scl"] = xr.Variable(
        #    dims=("time", "overpass", "lat", "lon"),
        #    data = scl_data_padded,
        #    attrs={"long_name": "Sentinel-2-L2A scene classification data", "units": "unitless"}
        #)

        times_clean = [l["launch_time"].astimezone(timezone.utc).replace(tzinfo=None) for l in launch_data]
        times_np    = np.array(times_clean, dtype="datetime64[ns]")

        # --- Construct Dataset ---
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time":      ("time",      times_np),
                "overpass":  ("overpass",  np.arange(max_overpasses)),
                "y":         ("y",         y_coords, {"long_name": "pixel row index", "units": "1"}),
                "x":         ("x",         x_coords, {"long_name": "pixel col index", "units": "1"}),
                "lat_range": ("bbox_lat",  lat_range, {"long_name": "bbox latitude range [min, max]",
                                                        "units": "degrees_north"}),
                "lon_range": ("bbox_lon",  lon_range, {"long_name": "bbox longitude range [min, max]",
                                                        "units": "degrees_east"}),
            },
            attrs={"Data_Attribution": f"Sentinel-2 data for Site: {loc_name}"}
            )
        
        ds.to_netcdf("download/"+filename)

if __name__ == "__main__":     
    
    #Loop over all years and run functions depending on input arguments.
    LaunchData = find_data()  
    LaunchData.data_search()
