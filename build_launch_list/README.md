# ARIA Project
## Unlocking the potential to quantify spacecraft pollution from space-based broadband imagers 

### build_launch_database.py

This file uses Jonathan's Space Report (https://planet4589.org/space/gcat/web/launch/index.html) to compile a list of spaceports and associated orbital launches. 
- Date range is 2013-01-01 to 2026-11-01, to cover all Sentinel-2, Landsat 8, TROPOMI, and GEMS overpasses.
- We exclude military launches, and only include orbital launches with a COSPAR designation. This excludes failed, uncatalogued, endoatmospheric, mesosphere, and suborbital launches, as well as pad explosions.
- Ocean/air-launched rockets are excluded.
- Suborbital launches from Blue Origin are retained.
- For more information on filtering, see https://planet4589.org/space/gcat/web/launch/lcols.html.

### sites.json
This file contains information about launches occuring at global spaceports. Each spaceport has an associated full name, latitude, longitude, and list of launch times in UTC.