import h5py
import os
from PIL import Image
import numpy

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
    hf.create_dataset('x', data=d1)   #Data images.
    hf.create_dataset('y', data=d2)   #Groundtruht image.
    hf.create_dataset('m', data=d3)   #Boundaries image.
    hf.close

# Sets the name of the .hdf5 file.
file_name = 'mydata.hdf5'

# File names of the data files.
data_files = ['top_mosaic_09cm_area15_0.5.jpg', 'top_mosaic_09cm_area15_0.5.jpg', 'top_mosaic_09cm_area15_0.5.jpg']

# File names of the groundtruth file.
groundtruth_file = 'kit_nord.png'

# File names of the boundaries file.
boundaries_file = 'top_mosaic_09cm_area15_0.5.jpg'

d1 = []

for i in range(len(data_files)):
    img = Image.open(data_files[i])
    #red, green, blue = img.split()
    #img_arr = numpy.array(red)
    img_arr = numpy.array(img)
    img_arr.reshape(img_arr.shape[1],img_arr.shape[0])
    img_arr.tolist()
    d1.append(img_arr)
    img.close()
        
groundtruth = Image.open(groundtruth_file)
d2 = numpy.array(groundtruth)
groundtruth.close()
  
boundaries = Image.open(boundaries_file)
d3 = numpy.array(boundaries)
boundaries.close()
    

raw2hdf5(file_name, d1, d2, d3)

# Testing:
#hf = h5py.File('mydata.hdf5', 'r')
#print(hf.keys())
#n1 = hf.get('x')
#n1 = numpy.array(n1)
#print(n1[1])
#hf.close()


