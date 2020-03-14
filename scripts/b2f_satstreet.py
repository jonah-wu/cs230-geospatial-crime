import os

# path to image folder, get all filenames on this folder
# and store it in the onlyfiles list


def bin_files(bin1, bin2, bin3, sat_path, street_path):
    satellite_path = sat_path
    sat19_files = [f for f in os.listdir(satellite_path) if os.path.isfile(os.path.join(satellite_path, f))]
    #print(onlyfiles)
    #print(bin1)
    streetview_path = street_path 
    street_files = [f for f in os.listdir(streetview_path) if os.path.isfile(os.path.join(streetview_path, f))]
    # print(bin1)
    # print(bin2)
    # print(bin3)
    
    sat19_copy_path = "../data/satellite_imgs/sat19_imgs_cpy"
    sat19_copy_files = [f for f in os.listdir(sat19_copy_path) if os.path.isfile(os.path.join(sat19_copy_path, f))]
    
    
    sat19_lis1files = []
    sat19_lis2files = []
    sat19_lis3files = []

    lis1files = []
    lis2files = []
    lis3files = []

#    print(street_files)
#    print(sat19_files)
    print("B1's length is")
    print(len(bin1))
    for j in bin1:
        if j + '_sat_19.jpg' in sat19_copy_files and j + '.jpg' in street_files:
            lis1files.append(j + '.jpg')
            sat19_lis1files.append(j + '_sat_19.jpg')
    print("B2's length is")
    print(len(bin2))
    for j in bin2:
        if j + '_sat_19.jpg' in sat19_copy_files and j + '.jpg' in street_files:
            lis2files.append(j + '.jpg')
            sat19_lis2files.append(j + '_sat_19.jpg')
    print("B3's length is")
    print(len(bin3))
    for j in bin3:
        if j + '_sat_19.jpg' in sat19_copy_files and j + '.jpg' in street_files:
            lis3files.append(j + '.jpg')
            sat19_lis3files.append(j + '_sat_19.jpg')


            #for j in bin1:
        #if j + '.jpg' in onlyfiles:
#lis1files.append(j+'.jpg')
#print(len(lis1files))


    # create three sub folders in mypath folder
    subfolder1 = os.path.join(satellite_path, "bin1")
    subfolder2 = os.path.join(satellite_path, "bin2")
    subfolder3 = os.path.join(satellite_path, "bin3")

    # check if they already exits to prevent error
    if not os.path.exists(subfolder1):
        os.makedirs(subfolder1)

    if not os.path.exists(subfolder2):
        os.makedirs(subfolder2)

    if not os.path.exists(subfolder3):
        os.makedirs(subfolder3)

    # move files to their respective sub folders
    print("This is sat19_list1files")
    print(len(sat19_lis1files))
    for i in sat19_lis1files:
        #print(i)
        source = os.path.join(satellite_path, i)
        destination = os.path.join(subfolder1, i)
        if os.path.isfile(source):
            os.rename(source, destination)
    print("This is sat19_list2files")
    print(len(sat19_lis2files))
    for i in sat19_lis2files:
        #print(i)
        source = os.path.join(satellite_path, i)
        destination = os.path.join(subfolder2, i)
        if os.path.isfile(source):
            os.rename(source, destination)
    print("This is sat19_list3files")
    print(len(sat19_lis3files))
    for i in sat19_lis3files:
        #print(i)
        source = os.path.join(satellite_path, i)
        destination = os.path.join(subfolder3, i)
        if os.path.isfile(source):
            os.rename(source, destination)

    #print(onlyfiles)
    #print(bin1)

    # print(bin1)
    # print(bin2)
    # print(bin3)
    # create three sub folders in mypath folder
    street_subfolder1 = os.path.join(streetview_path, "bin1")
    street_subfolder2 = os.path.join(streetview_path, "bin2")
    street_subfolder3 = os.path.join(streetview_path, "bin3")

    # check if they already exits to prevent error
    if not os.path.exists(street_subfolder1):
        os.makedirs(street_subfolder1)

    if not os.path.exists(street_subfolder2):
        os.makedirs(street_subfolder2)

    if not os.path.exists(street_subfolder3):
        os.makedirs(street_subfolder3)

    # move files to their respective sub folders
    print("This is lis1files")
    print(len(lis1files))
    for i in lis1files:
        #print(i)
        source = os.path.join(streetview_path, i)
        destination = os.path.join(street_subfolder1, i)
        if os.path.isfile(source):
            os.rename(source, destination)
    print("This is lis2files")
    print(len(lis2files))
    for i in lis2files:
        #print(i)
        source = os.path.join(streetview_path, i)
        destination = os.path.join(street_subfolder2, i)
        if os.path.isfile(source):
            os.rename(source, destination)
    print("This is lis3files")
    print(len(lis3files))
    for i in lis3files:
        #print(i)
        source = os.path.join(streetview_path, i)
        destination = os.path.join(street_subfolder3, i)
        if os.path.isfile(source):
            os.rename(source, destination)

    print(len(lis1files + lis2files + lis3files))
    print(len(sat19_lis1files + sat19_lis2files + sat19_lis3files))







