import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
#matplotlib inline
import seaborn as sns
from shapely.geometry import Point, Polygon
import numpy as np
import googlemaps
from collections import defaultdict
#Imports for K-means not using numpy -https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import pickle
from collections import Counter

# geoidx_to_crimecounts_pkl = open("../data/regionidx_to_crimecounts.pickle","rb")
# geoidx_to_crimecounts = pickle.load(geoidx_to_crimecounts_pkl)


def print_distribution(regions_to_crimes):
    crimenum_to_regions = {
    	1 : 0, 
        5 : 0,
        10 : 0,
        20: 0,
        40: 0,
        80: 0,
        160: 0,
        320: 0,
        640: 0,
        1280: 0,
        2560: 0  #the maximum
    }

    prev_num = 0
    for key,item in regions_to_crimes.items():
        for number in crimenum_to_regions.keys():
            if item <= number:
                crimenum_to_regions[number] += 1

    prev_item = 0
    for key, item in crimenum_to_regions.items():
        print("There were " + str(item - prev_item) + " regions that had fewer than or equal to " + str(key) + " crimes in our data")
        prev_item = item


def run_KMeans(geoidx_to_crimecounts):
	kmeans = KMeans(init = 'k-means++', n_clusters = 3)
	data = np.log1p(np.asarray(list(geoidx_to_crimecounts.values())).reshape(-1,1))
	print(len(data))
	print(np.sum(data))
	y = kmeans.fit_predict(data)
	print(Counter(y))
	# Map the geo_indices to the labels
	keys = list(geoidx_to_crimecounts.keys())
	values = y
	geoidx_to_label = dict(zip(keys, values))
	return geoidx_to_label

def min_max_of_class(idx_to_label, geoidx_to_crimecounts, num_of_clusters = 3):
	cluster_zero = []
	cluster_one = []
	cluster_two = []
	for x in range(num_of_clusters):
		for key, value in geoidx_to_crimecounts.items():
			if idx_to_label[key] == 0: # then region belongs to this cluster
				cluster_zero.append(value)
			elif idx_to_label[key] == 1:
				cluster_one.append(value)
			else:
				cluster_two.append(value)

	zero_min = min(cluster_zero)
	zero_max = max(cluster_zero)
	one_min = min(cluster_one)
	one_max = max(cluster_one)
	two_min = min(cluster_two)
	two_max = max(cluster_two)
	print(" The max of cluster0 is " + str(zero_max) + " and the min is " + str(zero_min))
	print(" The max of cluster1 is " + str(one_max) + " and the min is " + str(one_min))
	print(" The max of cluster2 is " + str(two_max) + " and the min is " + str(two_min))

	return [(zero_min, zero_max), (one_min, one_max), (two_min, two_max)]
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c='red')
# plt.show()

# from shapely import geometry
# p1 = geometry.Point(0,0)
# p2 = geometry.Point(1,0)
# p3 = geometry.Point(1,1)
# p4 = geometry.Point(0,1)

# pointList = [p1, p2, p3, p4, p1]

# poly = Polygon([[p.x, p.y] for p in pointList])


# square_list.append({
# 				'id': i,
# 				'geometry': polygon_geom

# 			})

# my_gdf = gpd.GeoDataFrame(square_list)