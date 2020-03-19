import os
from os import walk


mypath = "/Users/phil/Desktop/VITON-Sonder/data/test/cloth"


f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break



filename= open("test_pairs1.txt","w+")

for item in f:
    item2 = item[:-6]+ "_0.jpg"
    newItem  = item2 + " " + item + "\n"
    print(newItem)
    filename.write(newItem)

filename.close()
