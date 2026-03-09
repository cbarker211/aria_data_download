#from sentinelhub import SHConfig
import sys
import os
import numpy as np
from netCDF4 import Dataset
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
        self.target_res = [10]#, 60]

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

    def find_coincidences(self, bbox, deltat_minus, deltat_plus, launch_time, launch_data, scl_data_all, site):
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
        if len(results) == 0:
            return launch_data, scl_data_all
        
        # Duplicate tiles into unique overpasses by rounding to nearest minute
        unique_overpasses = {}
        for result in results:
            overpass_dt = datetime.fromisoformat(result['properties']['datetime'].replace("Z", "+00:00"))
            key = overpass_dt.replace(second=0, microsecond=0)
            if key not in unique_overpasses:
                unique_overpasses[key] = overpass_dt

        # Could we exploit multiple overpasses rather than only retaining one? 
        if len(unique_overpasses.keys()) > 1:
            print("Warning: Multiple overpasses found. Choosing closest.")

        # I get an error in the time format when the code is attempting to save the data to the NetCDF file.
        # There's a ".5" decimal value for some seconds entries that I suspect is causing the issue.
        # The error occured for me for multiple sites:  
        # KSC_LC39A, TYSC_LC9, JQ_LC43/94, JQLC43/91, CC_LC40, GIK-1_LC43/4, GIK-1_LC43/3, JQ_LC120, JQ_LC130 
        overpass_dt = min(unique_overpasses.values(), key=lambda dt: abs((dt - launch_time).total_seconds()))
        delta_t_mins = (overpass_dt - launch_time).total_seconds() / 60
        op_time_start = (overpass_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        op_time_end   = (overpass_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
        # https://sentiwiki.copernicus.eu/web/s2-processing
        # The classification map is produced for each Sentinel-2 Level-2A product at 20 m and resamples to 60 m resolution 
        # But we need to get the scl data at both 10m and 60m for output.
        for res in self.target_res:
            size = bbox_to_dimensions(bbox, resolution=res)

            # This is searching the level 2 (processed) data, which has the scene classification layer needed for cloud filtering.
            request_scl_data = SentinelHubRequest(
                data_folder='temp_dir', evalscript=self.evalscript_scl, bbox=bbox, size=size, config=self.config,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=self.config.sh_base_url),
                        time_interval=(op_time_start, op_time_end),
                        other_args={"id":"S2L2A"},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            )
            
            scl_data = request_scl_data.get_data()[0]
            if scl_data.size == 0:
                continue
            
            # Calculate cloud cover from scene classification data.
            cld_mask = np.isin(scl_data, [3, 8, 9, 10])
            cloud_prop = cld_mask.mean() * 100
            if cloud_prop > 20:
                continue
            
            if res == 10:
                print(f'Site = {site} on {launch_time}. Cloud-cover = {cloud_prop:.2f}%, ΔT = {delta_t_mins:.1f} mins.')
            
            launch_data[res].append((launch_time, cloud_prop, delta_t_mins))
            scl_data_all[res].append(scl_data)
        
        return launch_data, scl_data_all
    
    def retrieve_data(self, bbox, deltat_minus, deltat_plus, launch_data, scl_data_all, site, loc_name, lat_range, lon_range, scene):

        # Setup an output array to hold the data to save. (time, lat, lon, variable)
        sizes = {res: bbox_to_dimensions(bbox, resolution=res) for res in self.target_res}
        data_out = {res: np.full((len(launch_data[res]), sizes[res][1], sizes[res][0], 5), np.nan, dtype=np.float32) for res in self.target_res} 
        
        for res in self.target_res:

            if len(launch_data[res]) == 0:
                continue

            for launch_ind, (launch_time, _, delta_t_mins) in enumerate(launch_data[res]):

                if deltat_minus is None:
                    t_minus = delta_t_mins/60 - 1
                    t_plus  = delta_t_mins/60 + 1
                else:
                    t_minus = deltat_minus
                    t_plus  = deltat_plus

                time_start = (launch_time+timedelta(hours=t_minus)).strftime("%Y-%m-%dT%H:%M:%SZ")
                time_end   = (launch_time+timedelta(hours=t_plus)).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                request_res_dep_data = SentinelHubRequest(
                    data_folder='download',
                    evalscript=self.evalscript[res],
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=self.config.sh_base_url),
                            time_interval=(time_start, time_end),
                            other_args={"id":"S2L1C"},
                        )
                    ],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox,
                    size=sizes[res],
                    config=self.config,
                )

                # Get and check data:
                requested_data = request_res_dep_data.get_data()
                data_out[res][launch_ind,:,:,:] = requested_data[0]
                if len(requested_data) > 1:
                    raise ValueError('More than one image returned. Please check the data and code.')

                # Save an image of the launch site overpass for a visual check.
                fig, ax = plt.subplots(2,figsize=(15, 10),subplot_kw={"projection": ccrs.PlateCarree()})
                extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
                data = [requested_data[0][:,:,3], requested_data[0][:,:,4]]
                for i in range(2):
                    ax[i].imshow(data[i], extent=extent, origin="upper",transform=ccrs.PlateCarree())
                    ax[i].set_extent(extent, crs=ccrs.PlateCarree())
                plt.savefig(f'site_images/{site.replace("/","")}_{launch_time.strftime("%Y-%m-%d")}_{res}_{scene}.png')
                plt.close()

            # Save data for this site
            self.save_to_netcdf(site, res, sizes, data_out, launch_data, scl_data_all, lat_range, lon_range, loc_name, scene)
            print(f'{site}-{res} written to disk')

    def data_search(self):
        """This function downloads the Sentinel-2 data for the launch sites and times specified in the sites.json file."""

        # Open the sites file.
        site_data = json.load(open("sites.json"))
        sites = list(site_data.keys()) 
        sites = ["JQ_LC130"] 

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
            
            launch_data = {res: [] for res in self.target_res}
            scl_data_all = {res: [] for res in self.target_res}

            for time in tqdm(launch_times, total = len(launch_times), desc="Processing rows"):

                launch_time = datetime.fromisoformat(time).replace(tzinfo=timezone.utc)
                launch_data, scl_data_all = self.find_coincidences(bbox, 0, 1, launch_time, launch_data, scl_data_all, site)

            sites_data[site]["Targets"] = int(len(launch_data[self.target_res[0]]))
        
            self.retrieve_data(bbox, 0, 1, launch_data, scl_data_all, site, loc_name, lat_range, lon_range, "target")

            ############################################
            # Now look for overpasses before the launch. 
            ############################################

            # Only look either side of the launches we found.
            times_to_check = [t[0] for t in launch_data[10]]
            launch_data = {res: [] for res in self.target_res}
            scl_data_all = {res: [] for res in self.target_res}

            # The revisit time is latitude-dependent, so choose maximum 5 days either side.
            # We only find the nearest overpass anyway.
            for time in times_to_check:
                launch_data, scl_data_all = self.find_coincidences(bbox, -120, -5, time, launch_data, scl_data_all, site)
                if len(launch_data[10]) == 0:
                   launch_data, scl_data_all = self.find_coincidences(bbox, 5, 120, time, launch_data, scl_data_all, site)

            sites_data[site]["References"] = int(len(launch_data[self.target_res[0]]))
            
            self.retrieve_data(bbox, None, None, launch_data, scl_data_all, site, loc_name, lat_range, lon_range, "reference")
            # TODO: Sometimes no reference is found because the nearest overpasses are cloudy.
            # Need to have a loop to widen the search area.
            # Agreed. And consider relaxing the cloud cover threshold to 40 or 50, as the area around the launchpad might
            # be cloud free. I only obtained a reference and target scene for TYSC_LC16 and a ratio of these two scenes
            # yielded a small enhancement that looks like it might be a small plume. 

        df = pd.DataFrame.from_dict(sites_data, orient='index')
        df.index.name = 'Site'
        df.to_csv('coincidences.csv')

    def save_to_netcdf(self, site, res, sizes, data_out, launch_data, scl_data_all, lat_range, lon_range, loc_name, scene):

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

        data_vars = {}
        for ind, var in enumerate(variables[res]):
            data_vars[var] = xr.Variable(
                dims=("time", "lat", "lon"),
                data=data_out[res][:,:,:,ind],
                attrs={"long_name": long_names[res][ind], "units": units[ind]}
            )

        cloud_prop_all   = [l[1] for l in launch_data[res]]
        delta_t_mins_all = [l[2] for l in launch_data[res]]

        data_vars["cloud_prop"] = xr.Variable(
            dims=("time"),
            data=np.array(cloud_prop_all, dtype=np.float32),
            attrs={"long_name": "Cloud cover proportion in bbox", "units": "%"}
        )

        data_vars["delta_t_mins"] = xr.Variable(
            dims=("time"),
            data=np.array(delta_t_mins_all, dtype=np.float32),
            attrs={"long_name": "Time difference between overpass and launch", "units": "minutes"}
        )

        data_vars["scl"] = xr.Variable(
            dims=("time","lat", "lon"),
            data = np.stack(scl_data_all[res]),
            attrs={"long_name": "Sentinel-2-L2A scene classification data", "units": "unitless"}
        )

        y_coords  = np.arange(sizes[res][0], dtype=np.int32)
        x_coords  = np.arange(sizes[res][1], dtype=np.int32)
        times_clean = [t.astimezone(timezone.utc).replace(tzinfo=None)for t,_,_ in launch_data[res]]
        times_np  = np.array(times_clean, dtype="datetime64[ns]")

        # --- Construct Dataset ---
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time":      ("time",      times_np),
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
