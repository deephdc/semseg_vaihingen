# imports:
import h5py
import numpy as np
from os import path

import sys
import argparse
import semseg_vaihingen.config as cfg

from PIL import Image
# load jpeg or png image:
# use standard tools of Keras (skip cv2)
from keras import backend
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# obsolete, i.e not used
def rgb2gray(rgb):
    '''
    Function to convert RGB to gray using formula in 
    https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
    '''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_image_jpg(file_path):
    # set default dimension ordering as for TensorFlow
    backend.set_image_dim_ordering('tf')
    # load the image
    img = load_img(file_path)
    # convert to numpy array
    data = img_to_array(img, dtype='int')

    print("[DEBUG] data shape: {}".format(data.shape))
    print("[DEBUG] data {}".format(data[:10,:10,]))

    return data


# load one of the vaihingen images, specified by image_number; 
# by default only the first 3 channels are taken:
def load_vaihingen_image(filename, image_number, 
                         only_three_channels=True, show_properties=False,
                         convert_gray=True):
    debug = False
    # load the data and ground truth:
    f = h5py.File(filename)
    ground_truth = np.array(f['y_{}'.format(image_number)])
    ground_truth = np.transpose(ground_truth)
    data_raw = np.array(f['x_{}'.format(image_number)])
    data_raw = np.transpose(data_raw)
    f.close()

    # only use the first three channels:
    if only_three_channels:
        data_raw = data_raw[:, :, :3]
    
    if debug:
        print("[DEBUG] data shape: {}".format(data_raw.shape))
        print("[DEBUG] data {}".format(data_raw[:10,:10,]))
    
    if convert_gray:
        print("[INFO] Use conversation to grayscale!")
        img_bw = Image.fromarray(data_raw, 'RGB').convert('L')
        data_bw = img_to_array(img_bw, dtype='int')
        data = np.concatenate((data_bw, data_bw, data_bw), axis=2)
        if debug:
            print("[DEBUG] data_bw.shape: {}".format(data_bw.shape))
            print("[DEBUG] data {}".format(data[:10,:10,]))
    else:
        data = data_raw

    # show properties of the data and ground truth:
    if show_properties:
        print('- Ground truth:')
        print('type:  ', ground_truth.dtype)
        print('shape: ', ground_truth.shape)
        print('- Data:')
        print('type:  ', data.dtype)
        print('shape: ', data.shape)
        print('min:   ', np.min(data))
        print('max:   ', np.max(data))
        print('mean:  ', np.mean(data))
        print('dev:   ', np.sqrt(np.var(data)))

    return data, ground_truth


# overlap_factor < 1 means overlap, =1 means no overlapp, >1 empty space in between
def image_to_dataset(filename, image_number, window_shape, overlap_factor):
    # load the image and ground truth:
    data, ground_truth = load_vaihingen_image(filename, image_number)

    # iterate over the complete window to extract small patches:
    input_list = []
    output_list = []
    k = l = 0
    while k+window_shape[0] < data.shape[0]:
        k_tmp = k + window_shape[0]
        while l+window_shape[1] < data.shape[1]:
            l_tmp = l + window_shape[1]
            input_list.append(data[k:k_tmp, l:l_tmp, :])
            output_list.append(ground_truth[k:k_tmp, l:l_tmp])
            l += int(window_shape[1] * overlap_factor)
        k += int(window_shape[0] * overlap_factor)
        l = 0
    return input_list, output_list


# generate dataset consisting of 256x256 sized image patches (defined in config.py)
# spatial overlap of the patches can be specified
def generate_dataset(data_path, image_numbers, overlap_factor):
    # input / output size of the FCN:
    size = cfg.PATCH_SIZE

    # specify filename and directory:
    file_directory = data_path
    filename = 'vaihingen_'
    file_extension = '.hdf5'


    # create small batches of the complete images and save them in a list:
    x_list = []
    y_list = []
    for i in image_numbers:
        file = path.join(file_directory, filename + str(i) + file_extension)
        print(file)
        x, y = image_to_dataset(file, i, [size, size], overlap_factor)
        x_list.extend(x)
        y_list.extend(y)

    # convert the lists to 4D numpy arrays:
    num_samples = len(x_list)
    x_array = np.zeros((num_samples, size, size, 3), dtype=np.uint8)
    y_array = np.zeros((num_samples, size, size), dtype=np.uint8)
    for k in range(num_samples):
        x_array[k, :, :, :] = x_list[k]
        y_array[k, :, :] = y_list[k]
    print('Generated {} samples!'.format(num_samples))

    return x_array, y_array


# save the generatet training or validation set to .hdf5:
def save_dataset(name, x, y):
    f = h5py.File(name)
    f.create_dataset('x', data=x, dtype=np.uint8)
    f.create_dataset('y', data=y, dtype=np.uint8)
    f.close()


# used in other files to load the training and validation set:
def load_data(name):
    f = h5py.File(name)
    x = np.array(f['x'], dtype=np.float32)
    y = np.array(f['y'], dtype=np.uint8)
    print("[DEBUG] load_data, x.shape: {}".format(x.shape))
    return x, y


def main():
    
    data_path = args.input_patches_path
    output_path = args.datasets_path

    # files used for training:
    training_nums = [1, 3, 5, 7, 11, 13, 17, 21, 26, 28, 34, 37]

    # files used for validation:
    validation_nums = [30, 32]

    # generate and save the training and validation set:
    overlap = 0.6
    x_train, y_train = generate_dataset(data_path, training_nums, overlap)
    x_val, y_val = generate_dataset(data_path, validation_nums, overlap)

    save_dataset(path.join(output_path, cfg.TRAINING_DATA), x_train, y_train)
    save_dataset(path.join(output_path, cfg.VALIDATION_DATA), x_val, y_val)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_patches_path', type=str, required=True,
                        help='Location of the input image patches \
                        (e.g., /srv/semseg_vaihingen/data/raw/)')
    parser.add_argument('--datasets_path', type=str, required=True,
                        help='Location to write training and validation datasets \
                        (e.g., /srv/semseg_vaihingen/data/)')
    
    if len(sys.argv) != 3:
        print("")
        print("[ERROR] Wrong number of parameters! See usage:\n")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

###
#    if len(sys.argv)<2:
#      print ''
#      print '***********************************'
#      print 'Two parameters need to be specified:'
#      print '1. Location of the input image patches (e.g., /homea/hpclab/train001/data/vaihingen/ )'
#      print '2. Location to write training and validation sets (e.g., /homea/hpclab/train002/semseg/vaihingen/ )'
#      print '***********************************'
#
#      sys.exit()
###
    
    main()







