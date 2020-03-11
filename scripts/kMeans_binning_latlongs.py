import pandas as pd
import numpy as np
from collections import defaultdict

#Imports for the K-means- https://flothesof.github.io/k-means-numpy.html
#%matplotlib inline
#import matplotlib.pyplot as pyplot
#import seaborn as sns; sns.set()

#Imports for K-means not using numpy -https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import _kmeans

# -------------- Load pkl files ---------------- #
latlong_to_crimecounts_pkl = open("latlong_idx_to_crimecounts.pickle","rb")
latlong_to_crimecounts = pickle.load(latlong_to_crimecounts_pkl)


geoidx_to_crimecounts_pkl = open("geoindices_to_crimecounts.pickle","rb")
geoidx_to_crimecounts = pickle.load(geoidx_to_crimecounts_pkl)

latlongidx_to_geoidx = {}
latlong_to_regions = geo_grid.latlongs_to_regions(latlongs, geo_grid_df)
for key, idx in latlong_to_idx.items():
	geo_id = latlong_to_regions[key]
	latlongidx_to_geoidx.update({idx:geo_id})



kmeans = KMeans(init = 'k-means++', n_clusters = 3).fit(geoidx_to_crimecounts)




