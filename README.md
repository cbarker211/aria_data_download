# ARIA Project
## Unlocking the potential to quantify spacecraft pollution from space-based broadband imagers 

### build_launch_database.py

This file uses Jonathan's Space Report (https://planet4589.org/space/gcat/web/launch/index.html) to compile a list of spaceports and associated launches. Military, failed, suborbital, endoatmospheric, and mesopheric launches are excluded. Suborbital launches from Blue Origin are retained, as are launches with an uncertain timing (? in Launch_Date). For more information on filtering, see https://planet4589.org/space/gcat/web/launch/lcols.html.


### sites.json

This file contains information about launches occuring at global spaceports. Each spaceport has an associated full name, latitude, longitude, and list of launch times in UTC. 

### download_sentinel2_copernicus.py

This file uses the sentinelhub API to look for Sentinel-2 satellite overpasses within an hour of the launches detailed in sites.json. 