import pandas as pd
import numpy as np
from collections import defaultdict
import bin_to_folder as b2f

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

b1, b2, b3 = bin_imgs(indx_to_crimecounts)
data_path = "/Users/jonahwu/Documents/CSStanford/CS230/CS230Project/data/streetview_imgs"
b2f.bin_files(b1, b2, b3, data_path)

"""
kmeans = KMeans(init = 'k-means++', n_clusters = 3).fit(index_to_crimecounts)
print(kmeans)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c='red')
plt.show()
"""
