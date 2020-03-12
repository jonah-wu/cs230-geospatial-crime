import os

# path to image folder, get all filenames on this folder
# and store it in the onlyfiles list


def bin_files(bin1, bin2, bin3, path):
    mypath = path
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    #print(onlyfiles)
    #print(bin1)

    # print(bin1)
    # print(bin2)
    # print(bin3)
    
    if '6814.jpg' in onlyfiles:
        print('True')
    # create two seperate lists from onlyfiles list based on lis1 and lis2
    lis1files = [j + '_sat.jpg' for j in bin1 if j + '_sat.jpg' in onlyfiles]
    lis2files = [j + '_sat.jpg' for j in bin2 if j + '_sat.jpg' in onlyfiles]
    lis3files = [j + '_sat.jpg' for j in bin3 if j + '_sat.jpg' in onlyfiles]

    list1files19 = [j + '_sat_19.jpg' for j in bin1 if j + '_sat_19.jpg' in onlyfiles]
    list2files19 = [j + '_sat_19.jpg' for j in bin2 if j + '_sat_19.jpg' in onlyfiles]
    list3files19 = [j + '_sat_19.jpg' for j in bin3 if j + '_sat_19.jpg' in onlyfiles]


    lis1files += list1files19
    lis2files += list2files19
    lis3files += list3files19
        #for j in bin1:
        #if j + '.jpg' in onlyfiles:
#lis1files.append(j+'.jpg')
#print(len(lis1files))


    # create three sub folders in mypath folder
    subfolder1 = os.path.join(mypath, "bin1")
    subfolder2 = os.path.join(mypath, "bin2")
    subfolder3 = os.path.join(mypath, "bin3")

    # check if they already exits to prevent error
    if not os.path.exists(subfolder1):
        os.makedirs(subfolder1)

    if not os.path.exists(subfolder2):
        os.makedirs(subfolder2)

    if not os.path.exists(subfolder3):
        os.makedirs(subfolder3)

    # move files to their respective sub folders
    for i in lis1files:
        print(i)
        source = os.path.join(mypath, i)
        destination = os.path.join(subfolder1, i)
        os.rename(source, destination)

    for i in lis2files:
        print(i)
        source = os.path.join(mypath, i)
        destination = os.path.join(subfolder2, i)
        os.rename(source, destination)

    for i in lis3files:
        print(i)
        source = os.path.join(mypath, i)
        destination = os.path.join(subfolder3, i)
        os.rename(source, destination)


