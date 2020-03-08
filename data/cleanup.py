import os
import os.path as osp

data = "/Users/phil/Desktop/VITON-Sonder/data/test"


goodLines = []
with open("/Users/phil/Desktop/VITON-Sonder/data/test_pairs.txt", "r") as f:
    print(f)
    im_names = []
    c_names = []

    
    for line in f.readlines():
        linedata = line
        im_name, c_name = line.strip().split()
        im_names.append(im_name)
        c_names.append(c_name)

        path_i = osp.join(data, "image", im_name)
        path_c = osp.join(data, "cloth", c_name)

        print(path_i)
        print(path_c)
        if(os.path.exists(path_i) and os.path.exists(path_c)):
            goodLines.append(linedata)

#print(goodLines)

f= open("test_pairs1.txt","w+")

for item in goodLines:
     f.write(item)

f.close()


