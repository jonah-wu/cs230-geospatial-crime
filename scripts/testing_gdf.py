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


from shapely import geometry
p1 = geometry.Point(0,0)
p2 = geometry.Point(1,0)
p3 = geometry.Point(1,1)
p4 = geometry.Point(0,1)

pointList = [p1, p2, p3, p4, p1]

poly = Polygon([[p.x, p.y] for p in pointList])
print(poly)
square_list = []
square_list.append({
				'id': 1,
				'geometry': poly

			})

my_gdf = gpd.GeoDataFrame(square_list)
print(my_gdf.iloc[:1]['geometry'])
my_gdf.iloc[:1].plot()
print(my_gdf)
plt.show()