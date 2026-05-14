# =============================================================================
#  USGS/EROS Inventory Service Example
#  Python - JSON API
# 
#  Script Last Modified: 2/15/2023
#  Note: This example does not include any error handling!
#        Any request can throw an error, which can be found in the errorCode proprty of
#        the response (errorCode, errorMessage, and data properies are included in all responses).
#        These types of checks could be done by writing a wrapper similiar to the sendRequest function below
#  Usage: python download_data.py -u username -p token
# =============================================================================

# NOTE: The 'login' endpoint (used for authentication with the API) has been deprecated in February 2025. To continue using the API, use the 'login-token' endpoint for authentication.
#       Additional details can be found here: https://www.usgs.gov/media/files/m2m-application-token-documentation

import json
import requests
import sys
import time
import argparse
import threading
import re
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import rioxarray
from io import BytesIO
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from pyproj import Transformer


path = "" # Fill a valid path to save the downloaded files
maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
threads = []

# send http request
def sendRequest(url, data, apiKey = None):  
    pos = url.rfind('/') + 1
    endpoint = url[pos:]

    headers = {
        "Content-Type": "application/json"
    }
    
    if apiKey:
        headers = {'X-Auth-Token': apiKey} 
    response = requests.post(url, json=data, headers = headers)    
    
    output = json.loads(response.text)

    try:
      httpStatusCode = response.status_code 
      if response == None:
          print("No output from service")
          sys.exit()
      output = json.loads(response.text)	
      if output['errorCode'] != None:
          raise ConnectionError("Failed Request ID", output['requestId'], output['errorCode'], "-", output['errorMessage'])
      if  httpStatusCode == 404:
          raise ConnectionError("404 Not Found")
      elif httpStatusCode == 401: 
          raise ConnectionError("401 Unauthorized")
      elif httpStatusCode == 400:
          raise ConnectionError("Error Code", httpStatusCode)
    except Exception as e: 
          response.close()
          serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
          pos=serviceUrl.find('api')
          raise ConnectionError(f"Failed to parse request {endpoint} response. Re-check the input {json.dumps(data)}. The input examples can be found at {url[:pos]}api/docs/reference/#{endpoint}\n")
    response.close()    
    
    return output['data']

def search_landsat(launch_data, launch_data_dict, launch_lat, launch_lon, scene, cloud_cover, site, apiKey):

    #################################################################
    # First the search for matching scenes with minimal cloud cover. 
    #################################################################
    
    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"

    # Set up the acquisition filter.
    launch_time = launch_data_dict["launch_time"]
    time_start =  (launch_time + timedelta(hours=0)).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end   =  (launch_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    acquisitionFilter = {"start": time_start, "end": time_end}
    datasetName = "landsat_ot_c2_l1"

    # Set up the spatial filter.
    dellatlon = 0.15
    minlon = launch_lon - dellatlon
    maxlon = launch_lon + dellatlon
    minlat = launch_lat - dellatlon
    maxlat = launch_lat + dellatlon
    spatialFilter = {"filterType": "mbr",
                     "lowerLeft": {
                         "latitude": minlat,
                         "longitude": minlon
                         },
                     "upperRight": {
                         "latitude": maxlat,
                         "longitude": maxlon
                         }}
    
    # Set up the cloud filter.
    cloudCoverFilter = {"min": 0, "max": cloud_cover, "includeUnknown": True}
    
    maxResults = 10  
    payload = {"maxResults": maxResults,
               'datasetName' : datasetName, 
               'sceneFilter' : {'spatialFilter' : spatialFilter, 'acquisitionFilter' : acquisitionFilter, 'cloudCoverFilter' : cloudCoverFilter}}
    
    scenes = sendRequest(serviceUrl + "scene-search", payload, apiKey)

    # If we haven't set our maxResults high enough, then rerun. 
    while True:
        if int(scenes["totalHits"]) > maxResults:
            payload["maxResults"] = int(scenes["totalHits"]) + 5
            scenes = sendRequest(serviceUrl + "scene-search", payload, apiKey)
        else:
            break
    
    if len(scenes['results']) == 0:
        return launch_data

    def get_metadata_value(scene, field_name):
        for item in scene["metadata"]:
            if item["fieldName"] == field_name:
                return item["value"]
        return None

    for scene in scenes['results']:
        acq_start = get_metadata_value(scene, "Start Time")
        acq_stop = get_metadata_value(scene, "Stop Time")
        print(acq_start, acq_stop)

    ####################################
    # Now download the specific scenes.
    ####################################

    # 1. Use the download-options endpoint to work out where to download the data from.
    sceneIds = []
    for result in scenes['results']:
        # Add this scene to the list I would like to download
        sceneIds.append(result['entityId'])

    payload = {'datasetName' : "landsat_ot_c2_l1", 'entityIds' : sceneIds, "includeSecondaryFileGroups": True}        
    downloadOptions = sendRequest(serviceUrl + "download-options", payload, apiKey)

    # 2. Find the QA_PIXEL file, which contains the cloud cover information.
    downloads = []
    for product in downloadOptions:
        if product["available"] and product["productName"] == "Landsat Collection 2 Level-1 Band File":
            for subproduct in product["secondaryDownloads"]:
                if "_QA_PIXEL.TIF" in subproduct["displayId"]:
                    downloads.append({
                        "entityId": subproduct["entityId"],
                        "productId": subproduct["id"]
                    })
                    break   
        if downloads:
            break                 

    # 3. Download the QA_PIXEL file.
    payload = {'downloads' : downloads,
               'label' : datetime.now().strftime("%Y%m%d_%H%M%S")}
    requestResults = sendRequest(serviceUrl + "download-request", payload, apiKey) 
    downloads = requestResults['availableDownloads']

    for download in downloads:
        response = requests.get(download['url'])
        response.raise_for_status()
        da = rioxarray.open_rasterio(BytesIO(response.content), masked=True, chunks=True)

        transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
        minx, miny = transformer.transform(minlon, minlat)
        maxx, maxy = transformer.transform(maxlon, maxlat)

        cropped = da.rio.clip_box(minx, maxx, miny, maxy)
        qa_data = cropped.squeeze().values

        cld_mask = np.isin(qa_data, [3, 8, 9, 10])
        cloud_prop = cld_mask.mean() * 100
        if cloud_prop > cloud_cover:
            continue

        plt.plot(qa_data)
        plt.savefig("test.png")

    
    sys.exit()
    # 2. Figure out what items we actually want from the download link.
    downloads = []
    for product in downloadOptions:
        if product["available"] == True and product["productName"] == "Landsat Collection 2 Level-1 Product Bundle":
            downloads.append({
                "entityId": product["entityId"],
                "productId": product["id"]
            })
    
    # 3. If we found products, then download them.
    if downloads:
        payload = {'downloads' : downloads,
                   'label' : datetime.now().strftime("%Y%m%d_%H%M%S")}
        # Call the download to get the direct download urls
        requestResults = sendRequest(serviceUrl + "download-request", payload, apiKey) 
        downloads = requestResults['availableDownloads']

        dataset = xr.Dataset()

        for download in downloads:
            pass

        

        def process_download(download, minlon, minlat, maxlon, maxlat):

            response = requests.get(download['url'])
            response.raise_for_status()

            da = rioxarray.open_rasterio(
                BytesIO(response.content),
                masked=True,
                chunks=True
            )

            clipped = da.rio.clip_box(
                minx=minlon,
                miny=minlat,
                maxx=maxlon,
                maxy=maxlat,
                crs="EPSG:4326"
            )

            filename = download['url'].split("/")[-1]
            band = re.findall(r'_(.*?)\.TIF', filename)[-1]

            return band, clipped.squeeze()
        with ThreadPoolExecutor(max_workers=6) as executor:

            futures = [
                executor.submit(
                    process_download,
                    download,
                    minlon,
                    minlat,
                    maxlon,
                    maxlat
                )
                for download in downloads
            ]

            for band, clipped in results:
                dataset[band] = clipped
    
    print(dataset)