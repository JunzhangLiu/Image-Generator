# GAN-Image-Generator
GUI Feature:<br/>
The generator takes a vector of size 100 as input to produce an output image of size 128\*128. The 100 sliders corresponds to the 100 values.<br/>
<img src="./demo/a.gif"><br/>
<br/>
The "random" button once clicked will randomize all the sliders to produce a random image.</br>
<img src="./demo/b.gif"><br/>
<br/>
This repository uses WGAN to generate 128\*128 images. I use it to generate anime faces, but it can be used to generate any image if trained using proper datasets.<br/>
I used the dataset on https://www.kaggle.com/splcher/animefacedataset and wrote delete_duplicate.py to delete duplicated images found in the dataset.<br/>
Images after 1 epoch:<br/>
![0_0](/sample_image/0_0.jpg?raw=true)
![0_1](/sample_image/0_1.jpg?raw=true)
![0_2](/sample_image/0_2.jpg?raw=true)
![0_3](/sample_image/0_3.jpg?raw=true)<br/>
Images after 2 epoch:<br/>
![1_0](/sample_image/1_0.jpg?raw=true)
![1_1](/sample_image/1_1.jpg?raw=true)
![1_2](/sample_image/1_2.jpg?raw=true)
![1_3](/sample_image/1_3.jpg?raw=true)<br/>
Images after 12 epoch:<br/>
![12_0](/sample_image/12_0.jpg?raw=true)
![12_1](/sample_image/12_1.jpg?raw=true)
![12_2](/sample_image/12_2.jpg?raw=true)
![12_3](/sample_image/12_3.jpg?raw=true)<br/>
Images after 23 epoch:<br/>
![23_0](/sample_image/23_0.jpg?raw=true)
![23_1](/sample_image/23_1.jpg?raw=true)
![23_2](/sample_image/23_2.jpg?raw=true)
![23_3](/sample_image/23_3.jpg?raw=true)<br/>
Images after 130 epoch:<br/>
![130_0](/sample_image/130_0.jpg?raw=true)
![130_1](/sample_image/130_1.jpg?raw=true)
![130_2](/sample_image/130_2.jpg?raw=true)
![130_3](/sample_image/130_3.jpg?raw=true)<br/>
Images after 161 epoch:<br/>
![161_0](/sample_image/161_0.jpg?raw=true)
![161_1](/sample_image/161_1.jpg?raw=true)
![161_2](/sample_image/161_2.jpg?raw=true)
![161_3](/sample_image/161_3.jpg?raw=true)<br/>
