import pandas as pd
import geo_grid
import numpy as np
from collections import defaultdict
import bin_to_folder as b2f
import csv
import pickle
import kmeans_for_geo
#Imports for K-means not using numpy -https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
#from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.cluster import _kmeans
# Read in csv
latlong_df = pd.read_csv("../data/street_intersections.csv")
crime_df = pd.read_csv("../data/crime.csv")
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

# Takes in our crimecounts, and returns three lists of indices representing, low examples, medium examples, and high examples
def bin_imgs(indx_to_crimecounts):
    counts = list(indx_to_crimecounts.values())
    print(counts.count(0))
    bin1_threshold = np.percentile(counts, 40)
    bin2_threshold = np.percentile(counts, 70)
    print("The 33rd percentile of # of crimes is " + str(bin1_threshold))
    print("The 66th percentile of # of crimes is " + str(bin2_threshold))
    bin1 = []
    bin2 = []
    bin3 = []
    for indx,y in indx_to_crimecounts.items():
        if y <= bin1_threshold:
            bin1.append(str(indx))
        elif y <= bin2_threshold:
            bin2.append(str(indx))
        else:
            bin3.append(str(indx))
    #print(bin1)
    return bin1, bin2, bin3

# Takes in our crimecounts, and returns three lists of indices representing, low examples, medium examples, and high examples
def bin_imgs(indx_to_crimecounts):
    counts = list(indx_to_crimecounts.values())
    print(counts.count(0))
    bin1_threshold = np.percentile(counts, 40)
    bin2_threshold = np.percentile(counts, 70)
    print("The 33rd percentile of # of crimes is " + str(bin1_threshold))
    print("The 66th percentile of # of crimes is " + str(bin2_threshold))
    bin1 = []
    bin2 = []
    bin3 = []
    for indx,y in indx_to_crimecounts.items():
        if y <= bin1_threshold:
            bin1.append(str(indx))
        elif y <= bin2_threshold:
            bin2.append(str(indx))
        else:
            bin3.append(str(indx))
    #print(bin1)
    return bin1, bin2, bin3

# populate a dictionary of # of regions with given crime_counts
def print_distribution(regions_to_crimes):
    region_to_crimes = {
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

    for key,item in regions_to_crimes.items():
        for number in region_to_crimes.keys():
            if item <= number:
                region_to_crimes[item] += 1

    for key, item in region_to_crimes.items():
        print("There were " + str(item) + "number of regions that had fewer than or equal to " + str(key) + " crimes with our data")


# Finally, from our crime dataset, count number of crimes that have occurred in the list of latlongs documented and create mapping from indx to crime counts.
stepsize = .003
latlongs, indx_to_latlong = indx_to_latlong(latlong_df)


print("Loading the grid....")
geo_grid_df = geo_grid.create_grid(stepsize, '1k')
num_of_regions = geo_grid_df['id'].size
print("We have " + str(num_of_regions) + " grids.")
geo_ids_to_polygons = geo_grid.ids_to_polygons(geo_grid_df)

print("Syncing up latlongs to the grid regions ....")
# print(geo_ids_to_polygons)
latlongs_to_regions = geo_grid.latlongs_to_regions(latlongs, geo_grid_df) 

#print(latlongs_to_regions)

# w = csv.writer(open("latlongs_to_regions.csv", "w"))
# for key, val in latlongs.items():
#     w.writerow([key, val])
print("Enumerating crime per latlong ...")
latlong_to_crimes = latlong_to_crimecounts(latlongs, crime_df)
regions_to_crimes = geo_grid.regions_to_crimes(latlong_to_crimes, latlongs_to_regions, num_of_regions)


idx_to_crimecount_pickle = open("../data/regionidx_to_crimecounts.pickle", "wb")
pickle.dump(regions_to_crimes, idx_to_crimecount_pickle)
idx_to_crimecount_pickle.close()


print("Hooking up the latlong indices for our images to the geo indices ...")
latlong_to_idx = {v: k for k, v in indx_to_latlong.items()}
latlongidx_to_geoidx = {}
for key, idx in latlong_to_idx.items():
    if key in latlongs_to_regions: 
        geo_id = latlongs_to_regions[key]
        latlongidx_to_geoidx.update({idx:geo_id})
    else:
        print("Key error," + str(key) +  "not found in latlong_to_regions")      
#print(latlongidx_to_geoidx)

a = 0
for value in latlongidx_to_geoidx.values():
    if value == 0:
        a += 1
print('We have ' + str(a) + " latlongs that do not sync up with a geo_id out of " + str(len(list(latlongidx_to_geoidx.keys()))) + "latlongs")
print(" Running Kmeans on our data and clustering into 3 bins... ")
geoidx_to_label = kmeans_for_geo.run_KMeans()
kmeans_for_geo.min_max_of_class(geoidx_to_label, regions_to_crimes)


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

b0, b1, b2 = create_bins_from_clusters(geoidx_to_label, latlongidx_to_geoidx)
data_path = "../data/streetview_imgs"


b0_pkl = open("../data/b0_list.pickle", "wb")
pickle.dump(b0, b0_pkl)
b0_pkl.close()

b1_pkl = open("../data/b1_list.pickle", "wb")
pickle.dump(b1, b1_pkl)
b1_pkl.close()


b2_pkl = open("../data/b2_list.pickle", "wb")
pickle.dump(b2, b2_pkl)
b2_pkl.close()


b2f.bin_files(b0, b1, b2, data_path)

# idx_to_crimecounts = {}
# for key in latlong_to_crimes:
#     if key in latlongs:
#         idx = latlong_to_idx[key]
#         crimecount = latlong_to_crimes[key]
#         idx_to_crimecounts.update({idx:crimecount})




"""
kmeans = KMeans(init = 'k-means++', n_clusters = 3).fit(index_to_crimecounts)
print(kmeans)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c='red')
plt.show()
"""
