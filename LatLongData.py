import pandas as pd
import numpy as np

def getCoords():
    df = pd.read_csv("street_intersections.csv")
    df = df.sort_values(by=['the_geom']).drop_duplicates('the_geom')

    df['the_geom'] = df['the_geom'].str.extract(r'(-\d+.\d+ \d+.\d+)', expand=False)

    latlong_dict = {}
    indx = 1000
    for x in df['the_geom']:
        val_pair = x.split()
        val = val_pair[1] + "," + val_pair[0]
        latlong_dict.update({indx:val})
        indx += 1

    return latlong_dict



