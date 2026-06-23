# Earth Access for NASA Earth Data
import earthaccess
from datetime import datetime
import pandas as pd

# Log into earthaccess
earthaccess.login()

time_start = datetime.fromisoformat(site_data[site]['launch_times'][0]).strftime("%Y-%m-%d")
time_end = datetime.fromisoformat(site_data[site]['launch_times'][-1]).strftime("%Y-%m-%d")

# Get all the overpasses from the first to the last launch from this site.
results_1c = earthaccess.search_data(
    short_name = 'PACE_OCI_L1C_SCI',
    bounding_box = coords_wgs84,
    temporal = (time_start, time_end),
)

# Create a dataframe of the start and end times of the overpasses found.
midpoint = []
for result in results_1c:
    start = pd.to_datetime(result['umm']['TemporalExtent']['RangeDateTime']["BeginningDateTime"])
    end = pd.to_datetime(result['umm']['TemporalExtent']['RangeDateTime']["EndingDateTime"])
    midpoint.append(start + (end - start) / 2)
df = pd.DataFrame({"midpoint": midpoint})

# Now find matching overpasses for each launch time.
matching_launches = []
launch_times = pd.to_datetime(site_data[site]['launch_times'], utc=True)
for t in launch_times:
    matching = df.loc[(t <= df["midpoint"]) & (df["midpoint"] <= t + pd.Timedelta("1h"))]
    if len(matching) > 0:
        matching_launches.append(t)

print(matching_launches)