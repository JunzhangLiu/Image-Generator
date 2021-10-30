import os
from PIL import Image,ImageStat
import numpy
import cv2
#more than half of the images are duplicates of one another, they may influence the performance of the network
#this program removes duplicates in the dataset by deleting all the images that has the same sum over all channels
#has complexity of O(n), just with a very large n. Slow but only have to do it once.
stats = set()
for root,_,files in os.walk("./anime_face/images/"):
    for f in files:
        file_name = root+f
        print("loading data",f)
        img = Image.open(file_name)
        stat = tuple(ImageStat.Stat(img).sum)
        if stat in stats:
            print(file_name,"is a duplicate")
            os.remove(file_name)
        else:
            print(file_name,"is unique")
            stats.add(stat)
