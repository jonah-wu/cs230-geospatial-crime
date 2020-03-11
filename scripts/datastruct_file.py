import pandas as pd
import geo_grid
import numpy as np
from collections import defaultdict
import bin_to_folder as b2f
import csv
import pickle

"""
Script creates all the following data_structs and writes them to pickle files for later reference and usage . 
"""

# ----------- Data ---------------------- #
latlong_df = pd.read_csv("../data/street_intersections.csv")
crime_df = pd.read_csv("../data/crime_data.csv")
#crime_df = crime_df[crime_df['point'].notna()] #Remove nan values
latlong_df = latlong_df.drop_duplicates(subset=['Latitude', 'Longitude'])
stepsize = .003
geo_grid_df = geo_grid.create_grid(stepsize, '1k')
print("After rounding to 5 decimal places we have " + str(latlong_df['Latitude'].size) + " unique lat-long intersections.")
print("Current crime csv has " + str(crime_df['Latitude'].size) + " crimes recorded")

# ---------- Data Structs ---------- #
def main():
	latlongs, indx_to_latlongs = indx_to_latlong(latlong_df)
	latlong_to_indx = {v: k for k, v in indx_to_latlongs.items()}
	latlong_to_crimecount, high_crimes_dict = latlong_to_crimecounts(latlongs, crime_df)
	indx_to_crimecounts = {latlong_to_indx[key]:latlong_to_crimecount[key] for key in latlong_to_crimecount if key in latlongs}


	latlongs_to_geoindx = geo_grid.latlongs_to_regions(latlongs, geo_grid_df) 
	geoindx_to_regions = geo_grid.ids_to_polygons(geo_grid_df)
	georegions_to_crimecounts = geo_grid.georegions_to_crimes(latlong_to_crimecount, latlong_to_geoindx)

	# -----------------Write to Pickle Files-------------------- #
	idx_to_crimecount_pickle = open("latlong_idx_to_crimecounts.pickle", "wb")
	pickle.dump(indx_to_crimecounts, idx_to_crimecount_pickle)
	idx_to_crimecount_pickle.close()


	geoids_to_crimecount_pickle = open("geoindices_to_crimecounts.pickle", "wb")
	pickle.dump(georegions_to_crimecounts, geoids_to_crimecount_pickle)
	geoids_to_crimecount_pickle.close()


# ----------- Data Struct Gen Methods --------------- # 
# Returns an indx_to_lat_long dictionary and list of keys to map to crime df (latlong)
def indx_to_latlong(latlong_df):
    indx_to_latlong_total = {} #total crime count
    indx_to_latlong_high = {} #not currently being used
    latlongs = []
    
    indx = 1000
    for x, y in zip(latlong_df['Latitude'], latlong_df['Longitude']):
        val = str(x) + "," + str(y)
        latlongs.append(val)
        indx_to_latlong_total.update({indx:val})
        indx_to_latlong_high.update({indx:val}) 
        indx += 1
    return latlongs, indx_to_latlong_total


# Returns an latlong to crime_count dictionary
def latlong_to_crimecounts(latlongs, crime_df):
    num_matches = 0
    num_high_crimes = 0
    latlong_to_crimes_total  = {k : 0 for k in latlongs}
    high_crimes = ["ARSON", "ASSAULT", "MISSING PERSON", "SEX OFFENSES, FORCIBLE", "SEX OFFENSES, NON-FORCIBLE"]
    # "Sex offenses, Non-Forcible" are quite minimal in the database I was able to access, so it is okay if it is 
    # equal to 0.
    # Note: One problem with this database is that it seems to be lacking important crimes, such as homicides, 
    # in its database; it is also relatively inconsistent with other databases in terms of description.
    high_crimes_dict = {} #nested dictionary of high crime count - {latlong :  {crime1: # crime1, crime2: #crime2}}
    for k in latlongs: #Update the keys from the high_crimes list into the high_crimes_dict
        high_crimes_dict.update({k:{}}) #initialize keys to an empty dict
        for crime in high_crimes: 
            high_crimes_dict[k][crime] = 0 #Initialize the inner dictionary
    # Note: The latitude and longitude of the crimes from 2003 database is based on specific addresses instead of 
    # the intersection that it occured, so we will allow for finding the closest intersection to be able to 
    # extract more data.
    for x, y, z in zip(crime_df['Latitude'], crime_df['Longitude'], crime_df['Category']):
        val = str(x) + "," + str(y)
        
        # Problem is that will not filter out for invalid data
        num_matches += 1

        # I commented this out below so that it the program can run faster by 
        # implementing the min function in the for loop
        #latlong_to_crimes_total[val] += 1 #Adding count for the cumulative, no distinction between crime types
        
        for crime in high_crimes: #Add count to specific crime in dictionary
            if crime == z:
                 # This val update function will be used to find matching intersections by finding the most similar 
        # keys. We hope that this will get the appropriate intersection. The code was taken from the following:
        # https://www.geeksforgeeks.org/python-find-the-closest-key-in-dictionary/ and https://stackoverflow.com/questions/7934547/python-find-closest-key-in-a-dictionary-from-the-given-input-key
                val = min(latlong_to_crimes_total.keys(), key = lambda key: 
                    abs((float(key.split(',')[0]) - float(x)) + (float(key.split(',')[1]) - float(y))))
                high_crimes_dict[val][crime] += 1
                num_high_crimes += 1
                # These are print statements to check the contents of the dictionary
                print("Stats for " + val + ": ")
                print(high_crimes_dict[val])
                print("Num Crimes: " + str(num_matches))
                print("Num High Crimes: " + str(num_high_crimes))
    #Print statements
    print("Out of " + str(crime_df['Latitude'].size) + " crimes in our dataset, there were " + str(num_matches) + " matches")
    print("Out of " +  str(num_matches) + " matches we retrieved, there were " + str(num_high_crimes) + " high crimes")
    return latlong_to_crimes_total, high_crimes_dict


if __name__ == '__main__':
	main()

