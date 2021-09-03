from PyQt5 import QtWidgets as qw
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication as qa
from PyQt5.QtGui import QIcon, QPixmap,QImage
import sys
import numpy as np
from model import Model
import tensorflow as tf
from PIL import Image
import os
from PIL.ImageQt import ImageQt
import random

class Slider_callback():
    def __init__(self,arg,fun):
        self.arg = arg
        self.fun = fun
    def __call__(self):
        return self.fun(self.arg)


class GUI(QMainWindow):
    def __init__(self,x,y,wid,ht,model,generator_input=100):
        super(GUI,self).__init__()
        self.setGeometry(x,y,wid,ht)
        self.wid = wid
        self.ht=ht
        self.precision = 1000000
        
        self.num_sliders = 20
        self.noise_dim = generator_input
        self.noise = np.zeros((1,1,1,self.noise_dim))
        self.model = model
        self.max_val,self.min_val = 1.2,-1.2
        self.slider_movement=True
        self.init_ui()
        
    def init_ui(self):
        scale = 2
        self.get_labels()
        self.set_pix_map()
        self.init_button()
        self.init_sliders()
        self.show()
    def init_button(self):
        self.random = qw.QPushButton('random', self)
        self.random.move(1700,700)
        self.random.clicked.connect(self.random_img)
        
        self.reset = qw.QPushButton('reset', self)
        self.reset.move(1700,900)
        self.reset.clicked.connect(self.reset_sliders)
    
    def reset_sliders(self):
        self.slider_movement = False
        self.noise = np.zeros_like(self.noise)
        for i in range(self.noise_dim):
            self.main_sliders[i].setValue(0)
        self.slider_movement = True
        self.set_pix_map()

    def random_img(self):
        self.noise = np.random.uniform(-1,1,(1,1,1,self.noise_dim))
        self.slider_movement=False
        for i in range(self.noise_dim):
            self.main_sliders[i].setValue(self.noise[0,0,0,i]*self.precision)
        self.slider_movement = True
        self.set_pix_map()
    def set_pix_map(self):
        self.generated_img = model.generate(self.noise).numpy()*128+127.5
        img = Image.fromarray(self.generated_img.astype(np.int8)[0],'RGB')
        img = ImageQt(img)
        pixmap = QPixmap.fromImage(img)
        self.label.setPixmap(pixmap)

    def get_labels(self):
        self.label = qw.QLabel(self)
        self.label.move(1700,500)
        self.label.resize(128,128)

    def init_sliders(self,gap = 40):
        self.main_sliders = []
        self.callback_idx = []
        for i in range(5):
            for j in range(self.num_sliders):
                self.main_sliders.append(qw.QSlider(Qt.Horizontal,self))
                self.main_sliders[i*self.num_sliders+j].setGeometry(i*320, j*40, 300, 30)
                self.main_sliders[i*self.num_sliders+j].setMinimum(self.min_val*self.precision)
                self.main_sliders[i*self.num_sliders+j].setMaximum(self.max_val*self.precision)
                self.main_sliders[i*self.num_sliders+j].setValue(0)
                self.callback_idx.append(i*self.num_sliders+j)
                self.main_sliders[i*self.num_sliders+j].valueChanged.connect(Slider_callback(i*self.num_sliders+j,self.val_change))

    def val_change(self,idx):
        if self.slider_movement:
            self.noise[0,0,0,idx]=self.main_sliders[idx].value()/self.precision
            self.set_pix_map()
        
app = qa(sys.argv)
model = Model(2)
ckpt = tf.train.Checkpoint(model)
path = os.path.join(os.path.dirname(__file__),"saved_model/166/")
ckpt.read(path)#.assert_consumed()


gui = GUI(200,200,1500,900,model)
sys.exit(app.exec_())