# Build a database of all spaceports, each needs a lat/lon and list of datetimes for launches.

from datetime import datetime
import numpy as np
import argparse
import requests
from tqdm import tqdm
from web_scrape_func import scrape_jsr
import pandas as pd
import sys
import json

def clean_jsr_date(date):

    if pd.isna(date):
        return date

    date = date.strip()
    parts = date.split()

    # Remove values with "?" or only one part (e.g. "2015?")
    # We are checking overpass times, so really we need values with a time.
    if "?" in date:
        return "?"
    elif len(parts) < 4:
        return "Too short"
    else:
        time_part = parts[-1]

        # Case: HHMM:SS (e.g. 0947:10)
        if ":" in time_part and len(time_part) == 7:
            parts[-1] = f"{time_part[:2]}:{time_part[2:4]}:{time_part[5:]}"

        # Case: HHMM → 09:47
        elif time_part.isdigit() and len(time_part) == 4:
            parts[-1] = f"{time_part[:2]}:{time_part[2:]}:00"

    return " ".join(parts)

class import_launches:
    
    def __init__(self,start_year,final_year):
        """This function sets up the class.

        Args:
            year (_type_): The year.
        """        
        
        self.start_year = start_year
        self.final_year = final_year
        self.session = requests.Session()

    def get_launch_info_jsr(self):

        ####################################################################
        # Web scrape the launch data from Jonathan McDowell's JSR website.
        ####################################################################

        df       = scrape_jsr("https://planet4589.org/space/gcat/tsv/launch/launch.tsv",self.session)
        df_sites = scrape_jsr("https://planet4589.org/space/gcat/tsv/tables/sites.tsv",self.session)

        # Filter the launches to remove unwanted entries.
        df = df[df["#Launch_Tag"].str[:4].astype(int).between(self.start_year, self.final_year)]
        df = df.reset_index()
        print("Found",len(df),"launches from JSR for the years",self.start_year,"to",self.final_year)

        # Convert the launch date to a datetime, and filter out uncertain launches or launches with no timings.
        df["Launch_Date_raw"] = df["Launch_Date"].copy() # Save a copy of the raw date for debugging.
        df["Launch_Date"] = df["Launch_Date"].apply(clean_jsr_date) 
        uncertain_rows = len(df[df["Launch_Date"].isin(["?"])])
        no_time_rows   = len(df[df["Launch_Date"].isin(["Too short"])])
        df["Launch_Date"] = pd.to_datetime(df["Launch_Date"], errors="coerce") # Convert to datetime, setting errors to NaT (Not a Time) for failed parsing.
        df["Launch_Date"] = df["Launch_Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        bad_rows = len(df[df["Launch_Date"].isna()])
        df_stats = pd.DataFrame({
            "rows": ["Total Launches","Failed Date Parsing","Failed Uncertain","Failed No Time"],
            "vals": [len(df), bad_rows, uncertain_rows, no_time_rows],
            })
        print(df_stats)
        df = df.dropna(subset=["Launch_Date"])

        site_lookup = {}

        for _, row in df_sites.iterrows():
            site_code = row["#Site"]
            lat = row["Latitude"].strip()
            lon = row["Longitude"].strip()
            name = row["Name"].strip()

            site_lookup[site_code] = {
                "lat": lat,
                "lon": lon,
                "name": name
            }

        ####################################################################
        # Extract the required launch activity data and convert to a dict.
        ####################################################################

        # NOTE: should save the launch tag

        launch_sites = {}

        for site_code, group in df.groupby("Launch_Site"):

            if site_code not in site_lookup:
                continue

            launch_sites[site_code] = {
                "name": site_lookup[site_code]["name"],
                "lat":  site_lookup[site_code]["lat"],
                "lon":  site_lookup[site_code]["lon"],
                "launch_times": sorted(group["Launch_Date"])
            }

        with open("sites.json", "w") as f:
            json.dump(launch_sites, f, indent=4)

if __name__ == "__main__":
    """The main running of the program goes here. 
    All of the functions are inside the import_launches class.
    Call each of the functions using import_launches._function_
    e.g. import_launches.launches_per_year(start_year,end_year)
    """         
    
    # Set up the arguments for each function.
    parser = argparse.ArgumentParser()
    parser.add_argument('-sy',   "--start_year",   default = "2015", choices=str(np.arange(2015,2026)), help='Start Year (2015-2025).')
    parser.add_argument('-fy',   "--final_year",   default = "2025", choices=str(np.arange(2015,2026)), help='Final Year (2015-2025).')
    args = parser.parse_args()
    
    # Sort out the year range.
    start_year = int(args.start_year)
    final_year = int(args.final_year)
    if start_year > final_year:
        final_year = start_year + 1
    print(f"Processing from year {start_year} to {final_year}.")  
    
    #Loop over all years and run functions depending on input arguments.
    LaunchData = import_launches(start_year,final_year)  
    LaunchData.get_launch_info_jsr()