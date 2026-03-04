#from sentinelhub import SHConfig
import sys
import os
import numpy as np
from netCDF4 import Dataset
from sentinelhub.config import SHConfig
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub.geometry import BBox
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType, CRS
from sentinelhub.geo_utils import bbox_to_dimensions
from sentinelhub.api.process import SentinelHubRequest
import json
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
import xarray as xr
from tqdm import tqdm

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

        self.evalscript_60m_res = load_evalscript("evalscript_res", band="B01")
        self.evalscript_10m_res = load_evalscript("evalscript_res", band="B02")
        self.evalscript_scl     = load_evalscript("evalscript_scl")
    
    def download_data(self):
        """This function downloads the Sentinel-2 data for the launch sites and times specified in the sites.json file."""

        # Open the sites file.
        site_data = json.load(open("sites.json"))

        # Lets focus on Cape Canaveral for now, as this has the most launches.
        sites = ["CC"]
        #sites= list(site_data.keys())   

        # Define extent around target scene to sample
        # (balance between encompassing the scene and amount of data permitted for 10m resolution):
        dellat = 0.05
        dellon = 0.08

        # Target resolutions of output
        # (60 m for B01 or UB and 10 m for B02 or blue band):
        target_res = [10, 60]

        strband = ['B02', 'B01']
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
        
        # B01: ultrablue (442.7,442.3 nm) (S2A, S2B)
        # B02: blue (492.4 nm, 492.1 nm)

        # Loop over each site and each launch at that site, and look for Sentinel-2 data on that day.
        for site in sites:

            # Get the full spaceport name and list of launch times.
            loc_name = site_data[site]['name']
            launch_times = site_data[site]['launch_times'][40:41]
            print(f"Searching at {loc_name}.")

            # Set up grid box around location of interest:
            # TODO: We might need to specify the actual launch platform when there's multiple.
            loc_lat  = float(site_data[site]['lat'])
            loc_lon  = float(site_data[site]['lon'])
            lllon = loc_lon - dellon
            lllat = loc_lat - dellat
            urlon = loc_lon + dellon
            urlat = loc_lat + dellat
            coords_wgs84 = (lllon, lllat, urlon, urlat)
            bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
            lat_range = np.array([lllat, urlat], dtype=np.float64)  # [lat_min, lat_max]
            lon_range = np.array([lllon, urlon], dtype=np.float64)  # [lon_min, lon_max]

            processed_times = [{"dt":         datetime.fromisoformat(lt).replace(tzinfo=timezone.utc),
                                "time_start": datetime.fromisoformat(lt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "time_end":   (datetime.fromisoformat(lt).replace(tzinfo=timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "yyyymmdd":   datetime.fromisoformat(lt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d"),
                                } for lt in launch_times]

            # First check for which times have an overpass within 1 hour with low cloud cover.
            launch_times_success = {res: [] for res in target_res}
            for pt in tqdm(processed_times, desc="Processing rows"):

                launch_time = pt["dt"]
                time_start  = pt["time_start"]
                time_end    = pt["time_end"]
                yyyymmdd    = pt["yyyymmdd"]

                # First query if there's data on this day to minimise queries.
                search_iterator = self.catalog.search(DataCollection.SENTINEL2_L1C, bbox=bbox, time = (time_start, time_end))
                
                results = list(search_iterator)
                if len(results) == 0:
                    continue
                elif len(results) > 1:
                    print(results)
                    raise ValueError(f'One result expected for {yyyymmdd}. Please check the data and code.')
                
                # TODO: Is the overpass time in UTC?
                overpass_time = results[0]['properties']['datetime']
                overpass_dt = datetime.fromisoformat(overpass_time.replace("Z", "+00:00"))
                delta_t_mins = (overpass_dt - launch_time).total_seconds() / 60

                def check_overpass(resolution):

                    size = bbox_to_dimensions(bbox, resolution=resolution)

                    # We need this for cloud filtering and to save to netcdf.
                    request_scl_data = SentinelHubRequest(
                        data_folder='temp_dir',
                        evalscript=self.evalscript_scl,
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=self.config.sh_base_url),
                                time_interval=(time_start, time_end),
                                other_args={"id":"S2L2A"},
                                #mosaicking_order=MosaickingOrder.LEAST_CC,
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        bbox=bbox,
                        size=size,
                        config=self.config,
                    )
                    scl_data = request_scl_data.get_data()[0]
                    
                    # Calculate cloud cover from scene classification data.
                    cld_classes = [3, 8, 9, 10]
                    cld_mask = np.isin(scl_data, cld_classes)
                    cloud_prop = cld_mask.mean() * 100
                    if cloud_prop > 20:
                        return

                    print(f'Site = {site} on {yyyymmdd}. Cloud-cover = {cloud_prop:.2f}%, ΔT = {delta_t_mins:.1f} mins. Requesting {resolution}m data...')
                    launch_times_success[resolution].append(launch_time)
            
                for res in target_res:
                    check_overpass(res)
        
            # Setup an output array to hold the data to save. (time, lat, lon, variable)
            sizes = {res: bbox_to_dimensions(bbox, resolution=res) for res in target_res}
            data_out = {res: np.full((len(launch_times_success[res]), sizes[res][1], sizes[res][0], 5), np.nan, dtype=np.float32) for res in target_res} 
            
            # Now retrieve the actual data at resolution of interest.
            for res in target_res:

                for launch_ind, launch_time in enumerate(launch_times_success[res]):

                    time_start = launch_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    time_end   = (launch_time+timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    yyyymmdd   = launch_time.strftime("%Y-%m-%d")

                    if res == 10:
                        evalscript = self.evalscript_10m_res
                    elif res == 60:
                        evalscript = self.evalscript_60m_res
                    else:
                        raise ValueError('Resolution not recognized. Please choose either 10 or 60 m.')
                    
                    request_res_dep_data = SentinelHubRequest(
                        data_folder='download',
                        evalscript=evalscript,
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
                    if len(requested_data) > 1:
                        raise ValueError('More than one image returned. Please check the data and code.')
                    
                    data_out[res][launch_ind,:,:,:] = requested_data[0]

                    fig, ax = plt.subplots(2,figsize=(15, 10),subplot_kw={"projection": ccrs.PlateCarree()})
                    extent = [lllon, urlon, lllat, urlat]
                    data = [requested_data[0][:,:,3], requested_data[0][:,:,4]]
                    for i in range(2):
                        ax[i].imshow(data[i], extent=extent, origin="upper",transform=ccrs.PlateCarree())
                        ax[i].set_extent(extent, crs=ccrs.PlateCarree())
                    plt.savefig(f'site_images/{site}_{yyyymmdd}_{res}.png')
                    plt.close()

                filename = f'sentinel2-l1c-band-l2a-slc-{site}-{res}.nc'

                data_vars = {}
                for ind, var in enumerate(variables[res]):
                    data_vars[var] = xr.Variable(
                        dims=("time", "lat", "lon"),
                        data=data_out[res][:,:,:,ind],
                        attrs={"long_name": long_names[res][ind], "units": units[ind]}
                    )

                y_coords  = np.arange(sizes[res][0], dtype=np.int32)
                x_coords  = np.arange(sizes[res][1], dtype=np.int32)
                times_clean = [t.astimezone(timezone.utc).replace(tzinfo=None)for t in launch_times_success[res]]
                times_np  = np.array(times_clean, dtype="datetime64[ns]")

                # TODO: Output the SCL and cloud cover.
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
                print('===> ' + filename + ' written to disk')         

if __name__ == "__main__":     
    
    #Loop over all years and run functions depending on input arguments.
    LaunchData = find_data()  
    LaunchData.download_data()
