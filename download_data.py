from sentinelhub.geo_utils import bbox_to_dimensions
import numpy as np
import json
from datetime import datetime, timedelta, timezone
import xarray as xr
from tqdm import tqdm
from python_scripts.utils import build_grid
from python_scripts.copernicus import search_copernicus, download_copernicus
from python_scripts.m2m import search_landsat, sendRequest
from python_scripts.gee import search_gee
import argparse
import copy
import os
import sys
import ee

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

# Currently this searches for the launch overpass, and then searches for references before and after with minimal cloud cover.
# But we need all instances of cloud-free overpasses with the launchpad so we can gather multiple potential reference scenes.
# We then need to select reference overpasses that are most consistent with the target. 

# A potential approach could be to find these as cloud-free scenes, but stick with a 20% or even 40% threshold for the coincidences with rocket launches. 
# So 0% for references, and <20% for rockets?
class find_data:

    def __init__(self, platform):
        """This sets up the class"""

        self.platform = platform
        
        # Target resolutions of output
        if self.platform == "sn":
            # 60 m for B01 or UB (442.7,442.3 nm) (S2A, S2B)
            self.target_res = 60

        elif self.platform == "ls":
            #username = "connorbarker"
            #token = np.loadtxt("/home/ucfacb0/.config/.m2m", dtype=str).item()        
            #serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
            #payload = {'username' : username, 'token' : token}
            #self.apiKey = sendRequest(serviceUrl + "login-token", payload)  

            key_path = '/home/ucfacb0/.config/earthengine/satroc-ef1c505b5935.json'

            credentials = ee.ServiceAccountCredentials(
                email='ee-satroc@satroc.iam.gserviceaccount.com',
                key_file=key_path
            )

            ee.Initialize(credentials=credentials, project='satroc')

            ls8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
            ls9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
            self.landsat = ls8.merge(ls9)

    def target_main(self, site):
        """This function downloads data for the launch sites and times specified in the sites.json file."""

        # Open the sites file.
        site_data = json.load(open("sites.json"))

        # Make the blank coincidence file if it doesn't already exist.
        if f"matches_{self.platform}.json" in os.listdir("./matches"):
            site_data_coincidence = json.load(open(f"./matches/matches_{self.platform}.json"))
        else:
            site_data_coincidence = {}

        if site == "all":
            sites_to_process = site_data.keys()
        else:
            sites_to_process = [site]

        for site in sites_to_process:
            # Get the full spaceport name and list of launch times.
            loc_name       = site_data[site]['name']
            launch_metadata = list(zip(site_data[site]['launch_times'], site_data[site]['launch_rockets'], site_data[site]['launch_prop_mass_kg']))
            print(f"Searching at {loc_name}.")

            site_data_coincidence[site] = {'name': loc_name,
                                           'lat': site_data[site]["lat"],
                                           'lon': site_data[site]["lon"],
                                           'launch_times':         [],
                                           'launch_rockets':      [],
                                           'launch_prop_mass_kg': []}

            # Set up grid box around site:
            launch_lat = float(site_data[site]['lat'])
            launch_lon = float(site_data[site]['lon'])
            bbox, lat_range, lon_range, bbox_gee = build_grid(launch_lat, launch_lon)
            region = ee.Geometry.Rectangle(bbox_gee)

            # Check for which times have an overpass within 1 hour with low cloud cover.
            launch_data = []

            for i, (time, rocket, prop) in tqdm(enumerate(launch_metadata), total = len(launch_metadata), desc="Processing rows"):

                launch_time = datetime.fromisoformat(time).replace(tzinfo=timezone.utc)

                # Set up the dictionary to contain all the information about the overpasses found.
                launch_data_dict = {"launch_time": launch_time,
                                    "launch_rocket": rocket,
                                    "launch_prop": prop
                                    }

                ######################
                # Search for matches.
                ######################
                if self.platform == "ls":
                    #launch_data = search_landsat(launch_data, launch_data_dict, launch_lat, launch_lon, "tar", 10, site, self.apiKey)
                    #launch_data = search_copernicus(launch_data, launch_data_dict, bbox, "tar", 10, site, self.platform)
                    launch_data  = search_gee(launch_data, launch_data_dict, self.landsat, region, "tar", 10, site)

                elif self.platform == "sn":
                
                    launch_data = search_copernicus(launch_data, launch_data_dict, bbox, "tar", 10, site, self.platform)

                ###############################################
                # Write to matches file and download the data.
                ###############################################

                if len(launch_data) > 0:

                    if self.platform == "ls":       
                        pass
                    elif self.platform == "sn":
                        self.download(bbox, 0, 1, launch_data, site, loc_name, lat_range, lon_range, "tar")

            # Write matched launches into coincidence dict using original indices
            for entry in launch_data:
                site_data_coincidence[site]['launch_times'].append(entry["launch_time"].strftime("%Y-%m-%dT%H:%M:%S"))
                site_data_coincidence[site]['launch_rockets'].append(entry["launch_rocket"])
                site_data_coincidence[site]['launch_prop_mass_kg'].append(str(entry["launch_prop"]))

        # Remove sites where no coincidences were found.
        site_data_coincidence = {site: data
                                 for site, data in site_data_coincidence.items()
                                 if len(data["launch_times"]) != 0}
        # Save coincidences
        with open(f"matches/matches_{self.platform}.json", "w") as f:
            json.dump(site_data_coincidence, f, indent=4)

    def ref_main(self, site):
        """This function downloads the Sentinel-2 data for a launch site over a specified year."""

        # Open the sites file.
        site_meta = json.load(open("sites.json"))
        site_data = json.load(open(f"matches/matches_{self.platform}.json"))

        if site == "all":
            sites_to_process = site_data.keys()
        else:
            sites_to_process = [site]

        for site in sites_to_process:

            launch_years = [int(date[:4]) for date in site_data[site]['launch_times']]
            unique_launch_years = sorted(list(set(launch_years)))

            for year in unique_launch_years:
                self.year = year

                # Get the full spaceport name and list of launch times.
                loc_name = site_meta[site]['name']
                print(f"Searching at {loc_name}.")

                # Set up grid box around site:
                bbox, lat_range, lon_range, _ = build_grid(site_meta[site]['lat'], site_meta[site]['lon'])  
                time_start = datetime(self.year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                time_end   = datetime(self.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                ref_data = search_copernicus([], {}, time_start, time_end, bbox, "ref", 5, site)

                if len(ref_data) == 0:
                    raise ValueError("No overpasses found.")
                self.download(bbox, -1, 1, ref_data, site, loc_name, lat_range, lon_range, "ref")

    def download(self, bbox, deltat_minus, deltat_plus, data, site, loc_name, lat_range, lon_range, scene):

        # Data is a list where each item is a cloud_free overpass.

        # Build an array of dimensions (time, lat, lon, variable).
        self.width, self.height = bbox_to_dimensions(bbox, resolution=self.target_res)
        data_out = np.full((len(data), self.height, self.width, 17), np.nan, dtype=np.float32)
        
        # Loop over resolutions and images.
        for ind, image in enumerate(data):

            if scene == "tar":
                time = image["launch_time"]
            else:
                time = image["overpass_time"]
            print(time)
            time_start = (time+timedelta(hours=deltat_minus)).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_end   = (time+timedelta(hours=deltat_plus)).strftime("%Y-%m-%dT%H:%M:%SZ")

            if self.platform == "sn":
                data_out = download_copernicus(data_out, self.target_res, time, time_start, time_end, ind, lon_range, lat_range, bbox, site, scene)
            
        # Save data for this site
        self.save_to_netcdf(site, self.target_res, data_out, data, lat_range, lon_range, loc_name, scene)
        print(f'{site}-{self.target_res} written to disk')

    def save_to_netcdf(self, site, res, data_out, data, lat_range, lon_range, loc_name, scene):
        
        # Set up strings for saving to netCDF
        if scene == "tar":
            filename = f"sentinel2-l1c-band-l2a-slc-{site.replace('/', '')}-{res}-{scene}.nc"
        else:
            filename = f"sentinel2-l1c-band-l2a-slc-{site.replace('/', '')}-{res}-{scene}-{self.year}.nc"
        
        variables   = ["saa", "sza", "vaa", "vza", "dataMask", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "scl"]
        units       = ["degrees", "degrees", "degrees", "degrees", "unitless", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN","unitless"]
        long_names  = ['Sentinel-2-l1c solar azimuth angle', 
                       'Sentinel-2-l1c solar zenith angle',
                       'Sentinel-2-l1c viewing azimuth angle', 
                       'Sentinel-2-l1c viewing zenith angle',
                       'Sentinel-2-l1c data mask (0=nodata, 1=data)',
                       'Sentinel-2-l1c B01 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B02 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B03 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B04 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B05 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B06 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B07 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B08 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B09 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B11 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Sentinel-2-l1c B12 TOA reflectance in DN (DN=1e4*reflectance)',
                       'Scene classification mask (0 - No data, 1 - Saturated / Defective, 2 - Dark Area Pixels, 3 - Cloud Shadows, 4 - Vegetation, 5 - Bare Soils, 6 - Water, 7 - Clouds low probability / Unclassified, 8 - Clouds medium probability, 9 - Clouds high probability, 10 - Cirrus, 11 - Snow / Ice)']


        # Create arrays for all overpasses.
        cloud_prop = np.array([l["cloud_prop"] for l in data], dtype=np.float32)

        # Convert the data_out dict into xarray Variables with appropriate dimensions and attributes.
        data_vars = {}
        for ind, var in enumerate(variables):
            if data_out.ndim == 4:
                xrdata = data_out[..., ind]   # has variable axis
            else:
                xrdata = data_out 
            data_vars[var] = xr.Variable(
                dims=("time", "y", "x"),
                data=xrdata,
                attrs={"long_name": long_names[ind], "units": units[ind]}
            )

        # For the target scene, we also need delta_t_secs.
        if scene == "tar":
            times_clean = [l["launch_time"].astimezone(timezone.utc).replace(tzinfo=None) for l in data]
            delta_t_secs = np.array([l["delta_t_secs"] for l in data], dtype=np.float32)
            data_vars["delta_t_secs"] = xr.Variable(
                dims = ("time"),
                data = delta_t_secs,
                attrs= {"long_name": "Time difference between overpass and launch", "units": "seconds"}
            )
        else: 
            times_clean = [l["overpass_time"].astimezone(timezone.utc).replace(tzinfo=None) for l in data]
        times_np    = np.array(times_clean, dtype="datetime64[ns]")

        data_vars["cloud_prop"] = xr.Variable(
            dims = ("time"),
            data = cloud_prop,
            attrs= {"long_name": "Cloud cover proportion in bbox", "units": "%"}
        )

        # --- Construct Dataset ---
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time":      ("time",      times_np),
                "x":         ("x",         np.arange(self.width, dtype=np.int32), {"long_name": "pixel col index", "units": "1"}),
                "y":         ("y",         np.arange(self.height, dtype=np.int32), {"long_name": "pixel row index", "units": "1"}),
                "lat_range": ("bbox_lat",  lat_range, {"long_name": "bbox latitude range [min, max]",
                                                        "units": "degrees_north"}),
                "lon_range": ("bbox_lon",  lon_range, {"long_name": "bbox longitude range [min, max]",
                                                        "units": "degrees_east"}),
            },
            attrs={"Data_Attribution": f"Sentinel-2 data for Site: {loc_name}"}
            )
        
        if not os.path.exists(f"download/{site.replace('/', '')}"):
            os.makedirs(f"download/{site.replace('/', '')}")
        ds.to_netcdf(f"download/{site.replace('/', '')}/"+filename)

if __name__ == "__main__":

    # Set up the arguments for each function.
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt',  "--platform",    default = "sn",      help="Platform (Sentinel-2 - sn, Landsat - ls, PACE/OLI - pace).")
    parser.add_argument('-site',"--site",        default = "all",     help="Launch site.")
    parser.add_argument('-tar', "--save_target", action='store_true', help='Save target data for the year.')
    parser.add_argument('-ref', "--save_refs",   action='store_true', help='Save reference data for the year.')
    args = parser.parse_args() 
    
    #Loop over all years and run functions depending on input arguments.
    LaunchData = find_data(args.platform) 
    if args.save_target:
        LaunchData.target_main(args.site)
    if args.save_refs:
        LaunchData.ref_main(args.site)
