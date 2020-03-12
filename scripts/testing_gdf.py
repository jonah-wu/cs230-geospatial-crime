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
import geo_grid
from shapely import geometry
# p1 = geometry.Point(0,0)
# p2 = geometry.Point(1,0)
# p3 = geometry.Point(1,1)
# p4 = geometry.Point(0,1)

# pointList = [p1, p2, p3, p4, p1]

# poly = Polygon([[p.x, p.y] for p in pointList])
# print(poly)
# square_list = []
# square_list.append({
# 				'id': 1,
# 				'geometry': poly

# 			})

# my_gdf = gpd.GeoDataFrame(square_list)
# print(my_gdf.iloc[:1]['geometry'])
# my_gdf.iloc[:1].plot()
# print(my_gdf)
# plt.show()


#plt.rcParams["figure.figsize"] = [8,6]
#fig, ax = plt.subplots(figsize=(10, 10))
# Get the shape-file for NYC
zoningdistrict = GeoDataFrame.from_file('/Users/jonahwu/Desktop/SFZoningDistricts/geo_export_a5eea1cd-2a31-46ff-9b25-0342e8e1a5eb.shp')
print(zoningdistrict.head())
for col in zoningdistrict.columns: 
    print(col)
    print(zoningdistrict[col].size)


geo_df = geo_grid.create_grid(.002, "1k")
list_polygons = []
for index, row in geo_df.iterrows():
	list_polygons.append(row['geometry'])

gSeries_grid = GeoSeries(list_polygons)
print(gSeries_grid)


#fig, ax = plt.subplots(figsize = (8,6))
gSeries_grid.boundary.plot(color = 'black', zorder = 2)
#zoningdistrict = zoningdistrict.set_index('shape_area')
#zoningdistrict = zoningdistrict.sort_index()
zoningdistrict.plot(ax=ax, color='orange', zorder = 1)

plt.gca().set_xlim([-122.502, -122.376])
plt.gca().set_ylim([37.709, 37.810])

plt.show()





