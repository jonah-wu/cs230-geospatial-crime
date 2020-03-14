import pickle
import bin_to_folder as b2f
import b2f_satstreet 

b0_pkl = open("../data/b0_list.pickle","rb")
b0 = pickle.load(b0_pkl)

b1_pkl = open("../data/b1_list.pickle","rb")
b1 = pickle.load(b1_pkl)

b2_pkl = open("../data/b2_list.pickle","rb")
b2 = pickle.load(b2_pkl)

street_path = "../data/streetview_imgs"
#b2f.bin_files(b0, b1, b2, data_path)

sat_path = "../data/satellite_imgs/sat19_imgs"
#b2f.bin_files(b0, b1, b2, data_path)

b2f_satstreet.bin_files(b0,b1,b2, sat_path, street_path)
