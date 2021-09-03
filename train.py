import numpy as np
import tensorflow as tf
from model import *
from PIL import Image
import os

BATCH_SIZE = 64
NOISE_DIM = 100
load = 69
data = np.load("./anime_face/data_set.npy").astype(np.uint8)
data = data[:data.shape[0]//BATCH_SIZE*BATCH_SIZE]
model = Model(BATCH_SIZE,NOISE_DIM)
model.compile()
ckpt = tf.train.Checkpoint(model)
if load:
    ckpt.read(os.path.join(os.path.dirname(__file__),f"saved_model/{load:d}/"))
    model.discriminator_opt = keras.optimizers.Adam(learning_rate=1e-5,beta_1=0.)
    model.generator_opt = keras.optimizers.Adam(learning_rate=1e-5,beta_1=0.)
for i in range(0,200):
    np.random.shuffle(data)
    model.fit(data,batch_size=BATCH_SIZE,epochs=1,shuffle=True)
    print(i)
    imgs = model.generate(tf.random.uniform((10,1,1,NOISE_DIM),minval=-1, maxval=1)).numpy()
    for idx,img in enumerate(imgs):
        Image.fromarray((img*128 +127.5).astype(np.int8),'RGB').save("./generated_data/"+str(i)+"_"+str(idx)+".jpg")
    if i > 20:
        ckpt.write(os.path.join(os.path.dirname(__file__),f"saved_model/{i:d}/"))

