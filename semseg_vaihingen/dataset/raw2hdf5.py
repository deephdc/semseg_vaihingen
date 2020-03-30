import h5py
import os
from os import path
import semseg_vaihingen.config as cfg
import numpy

from PIL import Image

# One vaihingen dataset contains 3 groups: The data (x), groundtruth (y) and boundaries (m).
#
# The data contains 5 different types of images that are represented as a matrix of numbers:
#   - Near Infrared, Red, Green, Normalized, DSM and NVI.
# Each image must have the same height and width and consist of only one channel!
#
# The groundtruth contains only 1 matrix with the labels as numbers.
#
# The boundaries contains a matrix with 0's and 1's where 1' is the presence of a border.

def raw2hdf5(file_name, d1, d2, d3):
    file_name = file_name
    hf = h5py.File(file_name, 'w')
    hf.create_dataset('x_1', data=d1)   #Data images.
    hf.create_dataset('y_1', data=d2)   #Groundtruht image.
    hf.create_dataset('m_1', data=d3)   #Boundaries image.
    hf.close

# Sets the name of the .hdf5 file, notice that the number is important because the attr. depend on it.
file_name = 'dataexample_1.hdf5'

#Directory where the images are.
IMG_DIR = path.join(cfg.BASE_DIR,'data/examples/')

# File names of the data files.
data_files = ['vaihingen_x0.tif', 'vaihingen_x1.tif', 'vaihingen_x2.tif']

# File names of the groundtruth file.
groundtruth_file = 'vaihingen_y.tif'

# File names of the boundaries file.
boundaries_file = 'vaihingen_m.tif'

d1 = []

for i in range(len(data_files)):
    idir = path.join(IMG_DIR,data_files[i])
    img = Image.open(idir)
    img_arr = numpy.array(img)
    img_arr.reshape(img_arr.shape[1],img_arr.shape[0])
    img_arr.tolist()
    d1.append(img_arr)
    img.close()
        
groundtruth = Image.open(path.join(IMG_DIR,groundtruth_file))
d2 = numpy.array(groundtruth)
groundtruth.close()
  
boundaries = Image.open(path.join(IMG_DIR,boundaries_file))
d3 = numpy.array(boundaries)
boundaries.close()
    

raw2hdf5(file_name, d1, d2, d3)

# Testing:
#hf = h5py.File('mydata_1.hdf5', 'r')
#print(hf.keys())
#n1 = hf.get('x')
#n1 = numpy.array(n1)
#print(n1[1])
#hf.close()


