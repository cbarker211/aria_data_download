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

#print('Next: add scene classification data to 10m and 60m resolution output')

#from utils import plot_image

# Units:
# DN = 10000 * REFLECTANCE
# AOT = DN / 1000

# Jonathan's databases sometimes have multiple launch locations at the same spaceport, all with very similar lat/lon.
# E.g. "Cape Canaveral Air Station, Florida", "NASA John F. Kennedy Space Center, Florida", "Space Florida, Cape Canaveral"

# Maybe also download band B04 (665 nm) where NO2 absorption small
# (About 30-times less than at B01 and B02)

# Load configuration file
config = SHConfig()
#print(config)
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

catalog = SentinelHubCatalog(config=config)

# Table showing data available: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/

# Error check to make sure credientials have been provided:
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

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

evalscript_scl = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                datasource: "S2L2A",
                bands: ["SCL"],
                units: "DN"
            }],
            output: [{
                bands: 1,
                sampleType: "UINT8"
            }],
        };
    }

    function evaluatePixel(sample) {
        return [sample.SCL];
    }
"""

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

# Name the target scenes:
loc_name = ['Florida']
nsites = len(loc_name)
    
# Lat and long targets to loop over:
target_lats = [28.4600]
target_lons = [-80.5400]

# Define extent around target scene to sample
# (balance between encompassing the scene and amount of data permitted for
#  10m resolution):
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

# Loop over sites:
for w in range(nsites):

    # Track progress:
    print('Processing data for ' + loc_name[w])

    # Loop over resolutions:
    for i in range(len(target_res)):

        if i==0: continue

        # Track progress:
        print('Processing data at ' + strres[i])

        # Calculate lower-left and upper-right corner lats and longs:
        lllon = target_lons[w]-dellon
        lllat = target_lats[w]-dellat
        urlon = target_lons[w]+dellon
        urlat = target_lats[w]+dellat
        
        resolution = target_res[i]
        
        coords_wgs84 = (lllon, lllat, urlon, urlat)
        bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        #print(type(bbox))
        #print(coords_wgs84)
        #sys.exit()
        
        years = [2025]  #range(2014,2025,1)
        months = range(11,12,1)
        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]

        # Define output directory based on location and resolution:
        output_dir = os.path.join(loc_name[w], strres[i])

        # TODO: account for leap years (2016,2020,2024)

        for yy in years:

            stryr=str(yy)

            print('Year: ' + stryr)

            for mm in months:

                if (mm+1)<10:
                    strmon='0'+str(mm+1)
                else:
                    strmon=str(mm+1)

                print('Month: ' + strmon)

                # Define number of days to loop over based on month:
                ndays = days_in_month[mm]

                # Adjust if leap year and Feb:
                if strmon=='02' and (stryr=='2016' or stryr=='2020' or stryr=='2024'):
                    ndays = 29
                    print('leap year, so increasing no. of Feb days to 29')

                # Define days to loop over:
                days = range(0,ndays,1)

                for dd in days:
            
                    if (dd+1)<10:
                        strday='0'+str(dd+1)
                    else:
                        strday=str(dd+1)

                    print('Day: ' + strday)

                    # Define day in yyyy-mm-dd format:
                    yyyymmdd = (stryr+"-"+strmon+"-"+strday)
                    #yyyymmdd = '2020-06-03'

                    # First query if there's data on this day, then either
                    # skip code or proceed. This is done to avoid using up
                    # processing units and request quotas:
                    search_iterator = catalog.search(
                        DataCollection.SENTINEL2_L1C,  
                        bbox=bbox,
                        time=(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),    
                    )
                    
                    results = list(search_iterator)

                    # Skip if no Sentinel-2 data on this day:
                    if len(results)==0:
                        print('No data for '+yyyymmdd+'. Skipping.')
                        continue
                    else:
                        print('===> Data found for '+yyyymmdd)

                    # First get scene classification data at resolution of interest
                    # for cloud filtering.
                    request_scl_data = SentinelHubRequest(
                        data_folder='temp_dir',
                        evalscript=evalscript_scl,
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                                time_interval=(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),
                                other_args={"id":"S2L2A"},
                                #mosaicking_order=MosaickingOrder.LEAST_CC,
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        bbox=bbox,
                        size=size,
                        config=config,
                    )

                    scl_data = request_scl_data.get_data()
                    #print(np.nanmin(scl_data[0]),np.nanmax(scl_data[0]))

                    #print('Min, Max SCL: ', np.nanmin(output_data[0][:,:,0]),np.nanmax(output_data[0][:,:,0]))
                    #print('Min, Max target band: ', np.nanmin(output_data[0][:,:,3]),np.nanmax(output_data[0][:,:,3]))
                        
                    #sys.exit()

                    scl = scl_data[0]

                    #tot_pixels = (len())
                    # Get dimensions of data using sza:
                    imx = len(scl[:,0])
                    jmx = len(scl[0,:])

                    tot_pix = imx*jmx
                    #print(tot_pix)
                    cld_pix = np.where((scl==3)|(scl==8)|(scl==9)|(scl==10))[0]
                    cld_prop = 100*(float(len(cld_pix))/float(tot_pix))

                    if cld_prop > 20:
                        print('Too cloudy on '+yyyymmdd+'. Skipping.')
                        continue

                    print('===> Relatively cloud-free on '+yyyymmdd)

                    if resolution==10:
                        #print('Requesting 10m res')
                        request_res_dep_data = SentinelHubRequest(
                            data_folder='temp_dir',
                            evalscript=evalscript_10m_res,
                            input_data=[
                                SentinelHubRequest.input_data(
                                    data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                                    time_interval=(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),
                                    other_args={"id":"S2L1C"},
                                    #mosaicking_order=MosaickingOrder.LEAST_CC,
                                )
                            ],
                            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                            bbox=bbox,
                            size=size,
                            config=config,
                        )
                        
                    if resolution==60:
                        
                        #print('Requesting 60m res')
                        request_res_dep_data = SentinelHubRequest(
                            data_folder='temp_dir',
                            evalscript=evalscript_60m_res,
                            input_data=[
                                SentinelHubRequest.input_data(
                                    data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                                    time_interval=(yyyymmdd+"T00:00:00Z", yyyymmdd+"T23:59:59Z"),
                                    other_args={"id":"S2L1C"},
                                    #mosaicking_order=MosaickingOrder.LEAST_CC,
                                )
                            ],
                            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                            bbox=bbox,
                            size=size,
                            config=config,
                        )

                    #print(responses[0].headers)
                    #sys.exit()
                    #print("Processing units spent:", response[0].headers["x-processingunits-spent"])
                    #proc_units = responses[0].headers["x-processingunits-spent"]
                    #acc_proc_units += float(proc_units)

                    # Wait 55 sec if processing units exceeds allowable
                    # limit (300) in a minute:
                    #if acc_proc_units >= 280:
                    #    print('Pausing process, as PUs exceeded')
                    #    time.sleep(55)
                    #    # Reinitialize processing units:
                    #    acc_proc_units = 0

                    # Get and check data:
                    requested_data = request_res_dep_data.get_data()

                    # Extract relevant data:
                    sza = requested_data[0][:,:,0]
                    vza = requested_data[0][:,:,1]
                    datamask = requested_data[0][:,:,2]
                    toa_refl_target_band = requested_data[0][:,:,3]
                    toa_refl_b04 = requested_data[0][:,:,4]
                    #scl = scl_data[0]

                    # NEXT: save the data to disk in NetCDF format:
                    # Might need to figure out how to include lat and long information
                    # Add bbox information to global/metadata

                    #"sunZenithAngles", "viewZenithMean", "dataMask", "B02"

                    #optical_data = request_res_dep_data.get_data(save_data=True)
                    
                    #print(f"Returned data is of type = {type(optical_data)} and length {len(optical_data)}.")
                    #print(f"Single element in the list is of type {type(optical_data[-1])} and has shape {optical_data[-1].shape}")
            
                    #image = optical_data[0][:,:,3]
                    #print(f"Image type: {image.dtype}")

                    print('Stopped before save data, as this will trigger a folder/file not found error.')
                    sys.exit()

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
                    
                    l2a_scl = ncout.createVariable('scl', np.float32,
                                                   ('ncols','nrows'))
                    l2a_scl.units = 'unitless'
                    l2a_scl.long_name = 'Sentinel-2-l2a scene classification'
                    l2a_scl.short_name = 'SLC'
                    l2a_scl[:] = scl

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
            sys.exit()
