import pandas as pd
import geo_grid
import numpy as np
from collections import defaultdict
import bin_to_folder as b2f
import csv
import pickle
import kmeans_for_geo

"""
Script creates all the following data_structs and writes them to pickle files for later reference and usage . 
"""

# ----------- Data ---------------------- #
latlong_df = pd.read_csv("../data/street_intersections.csv")
#crime_df = pd.read_csv("../data/crime_data.csv")
#crime_df = crime_df[crime_df['point'].notna()] #Remove nan values
latlong_df = latlong_df.drop_duplicates(subset=['Latitude', 'Longitude'])
crime_pkl = open("../data/total_crimes_dict.pickle","rb")
latlong_to_crimecount = pickle.load(crime_pkl)
stepsize = .003
geo_grid_df = geo_grid.create_grid(stepsize, '1k')
print("After rounding to 5 decimal places we have " + str(latlong_df['Latitude'].size) + " unique lat-long intersections.")
num_of_crimes = np.sum(np.asarray(list(latlong_to_crimecount.values())))
print("Current crime csv has " + str(num_of_crimes) + " crimes recorded")

# ---------- Data Structs ---------- #
def main():

    latlongs, indx_to_latlongs = indx_to_latlong(latlong_df)
    latlong_to_indx = {v: k for k, v in indx_to_latlongs.items()}
    indx_to_crimecounts = {latlong_to_indx[key]:latlong_to_crimecount[key] for key in latlong_to_crimecount if key in latlongs}
    latlongs_to_geoid = geo_grid.latlongs_to_regions(latlongs, geo_grid_df) 
    geoids_to_regions = geo_grid.ids_to_polygons(geo_grid_df)
    num_of_regions = len(list(geoids_to_regions.keys()))
    geoid_to_crimecount = geo_grid.regions_to_crimes(latlong_to_crimecount, latlongs_to_geoid, num_of_regions)

    #print_distribution(geoid_to_crimecount)
    latlongid_to_geoid = {}
    for key, idx in latlong_to_indx.items():
        if key in latlongs_to_geoid: 
            geo_id = latlongs_to_geoid[key]
            latlongid_to_geoid.update({idx:geo_id})
        else:
            print("Key error," + str(key) +  "not found in latlong_to_regions")      

    print("Running KMeans on our data and clustering into 3 bins ...")
    geoid_to_label = kmeans_for_geo.run_KMeans(geoid_to_crimecount)
    cluster_min_maxes = kmeans_for_geo.cluster_min_maxes(geoid_to_label, geoid_to_crimecount)
    b0, b1, b2 = create_bins_from_clusters(geoid_to_label, latlongid_to_geoid)
    latlongidx_to_crimecounts = {}
    for key in latlong_to_crimecount:
        if key in latlongs:
            idx = latlong_to_indx[key]
            crimecount = latlong_to_crimecount[key]
            latlongidx_to_crimecounts.update({idx:crimecount})

    print('Adding in the leftover unpaired latlongs into the bins based on their raw crime count ...')
    b0_thresholds = cluster_min_maxes[0]
    b1_thresholds = cluster_min_maxes[1]
    b2_thresholds = cluster_min_maxes[2]
    for key, value in latlongidx_to_crimecounts.items():
        if value <= b0_thresholds[1] and value >= b0_thresholds[0]: # If less than b0's max threshold than 
            b0.append(str(key))
        elif value <= b1_thresholds[1] and value >= b1_thresholds[0]:
            b1.append(str(key))
        else:
            b2.append(str(key))


    print("Creating the pkle lists...")
    b0_pkl = open("../data/b0_list.pickle", "wb")
    pickle.dump(b0, b0_pkl)
    b0_pkl.close()

    b1_pkl = open("../data/b1_list.pickle", "wb")
    pickle.dump(b1, b1_pkl)
    b1_pkl.close()


    b2_pkl = open("../data/b2_list.pickle", "wb")
    pickle.dump(b2, b2_pkl)
    b2_pkl.close()

    print("Data finished processing")
	# -----------------Write to Pickle Files-------------------- #
	# idx_to_crimecount_pickle = open("latlong_idx_to_crimecounts.pickle", "wb")
	# pickle.dump(indx_to_crimecounts, idx_to_crimecount_pickle)
	# idx_to_crimecount_pickle.close()


	# geoids_to_crimecount_pickle = open("geoindices_to_crimecounts.pickle", "wb")
	# pickle.dump(georegions_to_crimecounts, geoids_to_crimecount_pickle)
	# geoids_to_crimecount_pickle.close()


# ----------- Data Struct Gen Methods --------------- # 
def create_bins_from_clusters(geoidx_to_label, latlongidx_to_geoidx):
    b0 = []
    b1 = []
    b2 = []
    for key, value in latlongidx_to_geoidx.items():
        if not value == 0: # Throw out latlongs that don't sync up to our regions
            label = geoidx_to_label[value] # the label
            if label == 0:
                b0.append(str(key)) # Append the latlong idx
            elif label == 1:
                b1.append(str(key))
            elif label == 2:
                b2.append(str(key))
    print("b0 has" + str(len(b0)) + "images in it")
    print("b1 has" + str(len(b1)) + "images in it")
    print("b2 has" + str(len(b2)) + "images in it")
    return b0, b1, b2






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


def print_distribution(regions_to_crimes):
    crime_to_regioncount = {
        5 : 0,
        10 : 0,
        25: 0,
        50: 0,
        100: 0,
        200: 0,
        500: 0,
        1000: 0,
        100000: 0 #the maximum
    }

    for key,value in regions_to_crimes.items():
        for number in crime_to_regioncount.keys():
            if value <= number:
                crime_to_regioncount[value] += 1

    for key, item in crime_to_regioncount.items():
        print("There were " + str(item) + "number of regions that had fewer than or equal to " + str(key) + " crimes with our data")






if __name__ == '__main__':
	main()

