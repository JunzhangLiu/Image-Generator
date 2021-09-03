import os
from PIL import Image,ImageStat
import numpy
import cv2
stats = {}
file_lists = []
#more than half of the images are duplicates of one another, they may influence the performance of the network
#this program removes duplicates in the dataset by deleting all the images that has the same sum
#slow but only have to do it once
for root,_,files in os.walk("./anime_face/images/"):
    for f in files:
        print("loading data",f)
        img = Image.open(root+f)
        stat = ImageStat.Stat(img)
        stats[f] = stat.sum

for i in files:
    for j in files:
        if i in stats and j in stats:
            if i == j:
                continue
            if stats[i]==stats[j]:
                stats.pop(j)
                print(i,"is the same as",j)
                os.remove(root+j)