import pandas as pd
import numpy as np
from collections import defaultdict

# Read in csv
latlong_df = pd.read_csv("street_intersections.csv")
crime_df = pd.read_csv("crime.csv")
crime_df = crime_df[crime_df['point'].notna()] #Remove nan values
latlong_df = latlong_df.drop_duplicates(subset=['Latitude', 'Longitude'])
print("After rounding to 5 decimal places we have " + str(latlong_df['Latitude'].size) + " unique lat-long intersections.")
print("Current crime csv has " + str(crime_df['Latitude'].size) + " crimes recorded")


# Regex Form of Extraction
# latlong_df['the_geom'] = latlong_df['the_geom'].str.extract(r'(-\d+.\d+ \d+.\d+)', expand=False)
# crime_df['point'] = crime_df['point'].str.extract(r'(-\d+.\d+ \d+.\d+)', expand=False)


# Returns an indx_to_lat_long dictionary and list of keys to map to crime df
def indx_to_latlong(latlong_df):
    indx_to_latlong = {}
    latlongs = []
    
    indx = 1000
    for x, y in zip(latlong_df['Latitude'], latlong_df['Longitude']):
        val = str(x) + "," + str(y)
        latlongs.append(val)
        indx_to_latlong.update({indx:val})
        indx += 1
    return latlongs, indx_to_latlong

# Returns an latlong to crime_count dictionary
def latlong_to_crimecounts(latlongs, crime_df):
    num_matches = 0
    latlong_to_crimes  = {k : 0 for k in latlongs}
    for x, y in zip(crime_df['Latitude'], crime_df['Longitude']):
        val = str(x) + "," + str(y)
        if val in latlong_to_crimes:
            num_matches += 1
            latlong_to_crimes[val] += 1
    print("Out of " + str(crime_df['Latitude'].size) + " crimes in our dataset, there were " + str(num_matches) + " matches")
    return latlong_to_crimes


# Finally, from our crime dataset, count number of crimes that have occurred in the list of latlongs documented and create mapping from indx to crime counts.
latlongs, indx_to_latlong = indx_to_latlong(latlong_df)
latlong_to_crimes = latlong_to_crimecounts(latlongs, crime_df)
latlong_to_indx = {v: k for k, v in indx_to_latlong.items()}

indx_to_crimecounts = {}
for key in latlong_to_crimes:
    if key in latlongs:
        indx = latlong_to_indx[key]
        crimecount = latlong_to_crimes[key]
        indx_to_crimecounts.update({indx:crimecount})
print(indx_to_crimecounts)


