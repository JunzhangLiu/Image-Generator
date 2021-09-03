import numpy as np
from PIL import Image,ImageOps
import os
# image = Image.open('anime_face/images/23247_2008.jpg')
# image.show()
# image=image.resize((128,128))
data_set = []
i = 0
for root,_,files in os.walk("./anime_face/images/"):
    for f in files:
        try:
            print("loading data",f)
            img = Image.open(root+f)
            img=img.resize((128,128))
            data = np.asarray(img)
            data_set.append(data)
            data_mirror = np.asarray(ImageOps.mirror(img))
            data_set.append(data_mirror)
            i+=1
        except Exception as e:
            print("failed to load file "+f,e)
print(len(data_set))
data_set = np.array(data_set).astype(np.uint8)
np.save("anime_face/data_set",data_set)