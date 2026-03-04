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
import calendar
import json
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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

##########################
# Set up Sentinel Hub API
##########################

# Load configuration file
config = SHConfig()
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
catalog = SentinelHubCatalog(config=config)

# Error check to make sure credientials have been provided:
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

#####################
# Define evalscripts 
#####################

#TODO: Next: add scene classification data to 10m and 60m resolution output'

evalscript_60m_res = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                datasource: "S2L1C",
                bands: ["sunZenithAngles", "viewZenithMean", "dataMask", "B01", "B04"],
                units: "DN"
            }],
            output: [{
                bands: 5,
                sampleType: "FLOAT32"
            }],
        };
    }

    function evaluatePixel(sample) {
        return [sample.sunZenithAngles, sample.viewZenithMean, sample.dataMask, sample.B01, sample.B04];
    }
"""

evalscript_10m_res = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                datasource: "S2L1C",
                bands: ["sunZenithAngles", "viewZenithMean", "dataMask", "B02", "B04"],
                units: "DN"
            }],
            output: [{
                bands: 5,
                sampleType: "FLOAT32"
            }],
        };
    }

    function evaluatePixel(sample) {
        return [sample.sunZenithAngles, sample.viewZenithMean, sample.dataMask, sample.B02, sample.B04];
    }
"""

# Jonathan's databases sometimes have multiple launch locations at the same spaceport, all with very similar lat/lon.
# E.g. "Cape Canaveral Air Station, Florida", "NASA John F. Kennedy Space Center, Florida", "Space Florida, Cape Canaveral"

# Open the sites file.
site_data = json.load(open("sites.json"))


# Lets focus on Cape Canaveral for now, as this has the most launches.
#sites = ["STB"]
sites= list(site_data.keys())    

# Define extent around target scene to sample
# (balance between encompassing the scene and amount of data permitted for 10m resolution):
dellat = 0.05
dellon = 0.08

# Target resolutions of output
# (60 m for B01 or UB and 10 m for B02 or blue band):
target_res = [10, 60]
# Define strings of these to save to disk:
strres = ['10m', '60m']
strband = ['B02', 'B01']

# B01: ultrablue (442.7,442.3 nm) (S2A, S2B)
# B02: blue (492.4 nm, 492.1 nm)

# Initialize processing units to track:
acc_proc_units = 0

# Instead of looping over every day and searching for Sentinel-2 data, we can loop over
# each site and each launch at that site, and look for Sentinel-2 data on that day.

# Loop over sites:
for site in sites:

    # Get the full spaceport name and list of launch times.
    loc_name = site_data[site]['name']
    launch_times = site_data[site]['launch_times']
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

    for launch_time in launch_times:

        # Define year, month, day of launch:
        launch_time = datetime.fromisoformat(launch_time).replace(tzinfo=timezone.utc)

        # Look over a window for the launch to 2 hours after.
        time_start = (launch_time).strftime("%Y-%m-%dT%H:%M:%SZ")
        time_end   = (launch_time+timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stryr  = str(launch_time.year)
        strmon = str(launch_time.month).zfill(2)
        strday = str(launch_time.day).zfill(2)
        # Define day in yyyy-mm-dd format:
        yyyymmdd = (stryr+"-"+strmon+"-"+strday)

        # Loop over resolutions:
        for i in range(len(target_res)):

            # Track progress:
            #print(f'Processing data for {loc_name} at {strres[i]}')

            resolution = target_res[i]
            size = bbox_to_dimensions(bbox, resolution=resolution)

            # Define output directory based on location and resolution:
            output_dir = os.path.join(loc_name, strres[i])

            # First query if there's data on this day, then either
            # skip code or proceed. This is done to avoid using up
            # processing units and request quotas:
            search_iterator = catalog.search(
                DataCollection.SENTINEL2_L1C,  
                bbox=bbox, 
                time = (time_start, time_end), #(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),    
                filter="eo:cloud_cover < 20",
            )
            
            results = list(search_iterator)

            # Skip if no clear-sky Sentinel-2 data on this day:
            if len(results) == 0:
                #print('No data for '+yyyymmdd+'. Skipping.')
                continue
            elif len(results) > 1:
                raise ValueError('More than one image returned. Please check the data and code.')

            cloud_cover = results[0]['properties']['eo:cloud_cover']    
            overpass_time = results[0]['properties']['datetime']
            overpass_dt = datetime.fromisoformat(overpass_time.replace("Z", "+00:00"))
            delta_t_mins = (overpass_dt - launch_time).total_seconds() / 60
            print(f'Site = {site} on {yyyymmdd}. Cloud-cover = {cloud_cover}, ΔT = {delta_t_mins:.1f} mins. Requesting {resolution}m data...')

            if resolution == 10:
                evalscript = evalscript_10m_res
            elif resolution == 60:
                evalscript = evalscript_60m_res
            else:
                raise ValueError('Resolution not recognized. Please choose either 10 or 60 m.')
            
            request_res_dep_data = SentinelHubRequest(
                data_folder='download',
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                        time_interval=(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),
                        other_args={"id":"S2L1C"},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=size,
                config=config,
            )

            # Get and check data:
            requested_data = request_res_dep_data.get_data()
            if len(requested_data) > 1:
                raise ValueError('More than one image returned. Please check the data and code.')

            # Extract relevant data:
            sza = requested_data[0][:,:,0]
            vza = requested_data[0][:,:,1]
            datamask = requested_data[0][:,:,2]
            toa_refl_target_band = requested_data[0][:,:,3]
            toa_refl_b04 = requested_data[0][:,:,4]

            fig, ax = plt.subplots(2,figsize=(15, 10),subplot_kw={"projection": ccrs.PlateCarree()})
            extent = [lllon, urlon, lllat, urlat]
            data = [toa_refl_target_band, toa_refl_b04]
            for i in range(2):
                ax[i].imshow(data[i], extent=extent, origin="upper",transform=ccrs.PlateCarree())
                ax[i].set_extent(extent, crs=ccrs.PlateCarree())
            plt.savefig(f'{site}_{yyyymmdd}_{resolution}.png')
            plt.close()

            # NEXT: save the data to disk in NetCDF format:
            # Might need to figure out how to include lat and long information
            # Add bbox information to global/metadata

            #"sunZenithAngles", "viewZenithMean", "dataMask", "B02"

            #optical_data = request_res_dep_data.get_data(save_data=True)
            
            #print(f"Returned data is of type = {type(optical_data)} and length {len(optical_data)}.")
            #print(f"Single element in the list is of type {type(optical_data[-1])} and has shape {optical_data[-1].shape}")
    
            #image = optical_data[0][:,:,3]
            #print(f"Image type: {image.dtype}")

            save_data = False
            if save_data:
                # Define new directory:
                new_dir = os.path.join('.',output_dir, stryr)

                if not os.path.exists(os.path.join(new_dir)):
                    os.makedirs(os.path.join(new_dir))

                filename = 'sentinel2-l1c-band-l2a-slc-'+yyyymmdd+'.nc'

                ncout = Dataset((new_dir+'/'+filename), mode='w',
                                format='NETCDF4')

                ncout.Data_Attribution = ("Sentinel-2 data for "+
                                        str(coords_wgs84)+" domain at "+
                                        strres[i]+". "+
                                        str(int(np.round(cld_prop)))+"% "+
                                        "affected by clouds.")
                #ncout.Source1 = "Description and url  Source one: Description and url"
                #ncout.Source2 = "Description and url  Source two: Description and url"

                ncout.createDimension('ncols', imx)
                ncout.createDimension('nrows', jmx)

                l1c_mask = ncout.createVariable('mask', np.float32,
                                                ('ncols','nrows'))
                l1c_mask.units = 'unitless'
                l1c_mask.long_name = 'Sentinel-2-l1c data mask (0=nodata, 1=data)'
                l1c_mask.short_name = 'dataMask'
                l1c_mask[:] = datamask

                l1c_sza = ncout.createVariable('sza', np.float32,
                                            ('ncols','nrows'))
                l1c_sza.units = 'degrees'
                l1c_sza.long_name = 'Sentinel-2-l1c solar zenith angle'
                l1c_sza.short_name = 'SZA'
                l1c_sza[:] = sza

                l1c_vza = ncout.createVariable('vza', np.float32,
                                            ('ncols','nrows'))
                l1c_vza.units = 'degrees'
                l1c_vza.long_name = 'Sentinel-2-l1c viewing zenith angle'
                l1c_vza.short_name = 'VZA'
                l1c_vza[:] = vza

                l1c_band = ncout.createVariable(strband[i], np.float32,
                                                ('ncols','nrows'))
                l1c_band.units = 'DN'
                l1c_band.long_name = ('Sentinel-2-l1c '+strband[i]+
                                    ' TOA reflectance in DN (DN=1e4*reflectance)')
                l1c_band.short_name = strband[i]
                l1c_band[:] = toa_refl_target_band

                l1c_b04 = ncout.createVariable('B04', np.float32,
                                                ('ncols','nrows'))
                l1c_b04.units = 'DN'
                l1c_b04.long_name = ('Sentinel-2-l1c B04 TOA reflectance'+
                                    ' in DN (DN=1e4*reflectance)')
                l1c_b04.short_name = 'B04'
                l1c_b04[:] = toa_refl_b04

                ncout.close()
                
                print('===> ' + filename + ' written to disk')
