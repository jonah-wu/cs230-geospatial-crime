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
#from datetime import datetime
# plt.rcParams["figure.figsize"] = [8,6]

# # Get the shape-file for NYC

# boros = GeoDataFrame.from_file('./SFZoningDistricts/geo_export_a5eea1cd-2a31-46ff-9b25-0342e8e1a5eb.shp')
# print(boros)
# # boros = boros.set_index('zonin')
# boros = boros.sort_index()
# print(boros)
# # Plot and color by borough
# boros.plot(column = 'zoning_sim')
# print(boros.geometry)

# # Get rid of are that you aren't interested in (too far away)
# # plt.gca().set_xlim([-122.502, -122.376])
# # plt.gca().set_ylim([37.709, 37.810])


# make a grid of latitude-longitude values
def create_grid(stepsize, place):
	xmin, xmax, ymin, ymax = 37.709, 37.810, -122.502, -122.376
	if place == '10k':
		#print(stepsize/.0001)
		stepsize_multiplier = 10000
		multiplier = 10000
	elif place == '100k':
		stepsize_multiplier = 100000
		multiplier = 100000
	elif place == '1k':
		stepsize_multiplier = 1000
		multiplier = 1000
	else: # default don't know what to do
	 	stepsize_multipler = 10000
	 	multiplier = 10000
	 	print("I'm sorry this seems to be an invalid stepsize, you should only take stepsizes in the order of ten-thousands or hundred-thousands place")
	square_list = []
	i = 1
	for x in range(int(xmin*multiplier), int(xmax*multiplier), int(stepsize*stepsize_multiplier)):
		for y in range(int(ymin*multiplier), int(ymax*multiplier), int(stepsize*stepsize_multiplier)):
			real_x = x / multiplier
			real_y = y / multiplier
			x_list = [real_x, real_x+stepsize, real_x+stepsize, real_x]
			y_list = [real_y, real_y, real_y+stepsize, real_y+stepsize]
			# if i < 100:
				# print(x_list)
				# print(y_list)
			polygon_geom = Polygon(zip(x_list, y_list))
			square_list.append({
				'id': i,
				'geometry': polygon_geom

			})
			i += 1

	my_gdf = gpd.GeoDataFrame(square_list)
	my_gdf.plot()
	my_gdf.head()
	# print(type(my_gdf))
	# breakpoint()
	#plt.show()
	return my_gdf



def ids_to_polygons(my_gdf):
	ids_to_polygon = {}
	for x in range(len(my_gdf)):
		ids_to_polygon.update({my_gdf['id'][x]:my_gdf['geometry'][x]})
	return ids_to_polygon
# print(create_grid(.002, '1k'))

#Create a mapping of geo_grids to the latlongs within that geo_grid
# Returns a mapping of the actual latlongs to their region_id in which they are located.
def latlongs_to_regions(latlongs, geo_grid_df):
	latlong_to_regions = {k : 0 for k in latlongs} # initializes the dictionary mapping
	a = 0
	for latlong in latlongs:
		p = latlong.split(',')
		point = Point(float(p[0]), float(p[1]))
		# print(point)

		geo_grid_df['in_region'] = geo_grid_df.geometry.apply(lambda x: point.within(x))
		row = geo_grid_df[geo_grid_df['in_region'] == True]

		if not row.empty:
			latlong_to_regions[latlong] = row['id'].values.tolist()[0]
			#print(latlong_to_regions[latlong])
			#print(latlong + " fits into a polygon!")
			a += 1
		#else:
			#print(latlong + "does not fit into a polygon!")
	print("I found that there were" + str(a) + "matches to the regions")
	return latlong_to_regions

def regions_to_crimes(latlongs_to_crimes, latlong_to_regions, num_of_regions):
	regions_to_crimes = {x : 0 for x in range(1, num_of_regions+1)}
	for key, item in latlongs_to_crimes.items():
		if key in latlong_to_regions:
			region_id = latlong_to_regions[key] # identify the region the latlong with documented crimes belongs to
			#print(region_id)
			if region_id in regions_to_crimes:
				regions_to_crimes[region_id] += item
	
	#print(regions_to_crimes)
	return regions_to_crimes

# xx, yy = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
# xc = xx.flatten()
# yc = yy.flatten()

if __name__ == '__main__':
	create_grid(.002, '1k')


# # Now convert these points to geo-data
# pts = GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
# in_map =  np.array([pts.within(geom) for geom in boros.geometry]).sum(axis=0)
# pts = GeoSeries([val for pos,val in enumerate(pts) if in_map[pos]])
# print(pts)


# # Plot to make sure it makes sense:
# pts.plot(markersize=1)

# # Now get the lat-long coordinates in a dataframe
# coords = []
# for n, point in enumerate(pts):
#     coords += [','.join(__ for __ in _.strip().split(' ')[::-1]) for _ in str(point).split('(')[1].split(')')[0].split(',')]