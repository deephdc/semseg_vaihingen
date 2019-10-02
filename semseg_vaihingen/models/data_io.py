# imports:
import h5py
import sys
import numpy as np
from sys import argv
from os import path

# load one of the vaihingen images, specified by image_number; by default only the first 3 channels are taken:
def load_vaihingen_image(filename, image_number, only_three_channels=True, show_properties=False):
    # load the data and ground truth:
    f = h5py.File(filename)
    ground_truth = np.array(f['y_{}'.format(image_number)])
    ground_truth = np.transpose(ground_truth)
    data = np.array(f['x_{}'.format(image_number)])
    data = np.transpose(data)
    f.close()

    # only use the first three channels:
    if only_three_channels:
        data = data[:, :, :3]

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


# generate dataset consisting of 256x256 sized image patches; spatial overlap of the patches can be specified
def generate_dataset(data_path, image_numbers, overlap_factor):
    # input / output size of the FCN:
    size = 256

    # specify filename and directory:
    file_directory = data_path
    filename = 'vaihingen_'
    file_extension = '.hdf5'


    # create small batches of the complete images and save them in a list:
    x_list = []
    y_list = []
    for i in image_numbers:
        file = path.join(file_directory, filename + str(i) + file_extension)
        print file
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
    return x, y


def main(arguments):
    
    data_path = arguments[1]
    output_path = arguments[2]

    # files used for training:
    training_nums = [1, 3, 5, 7, 11, 13, 17, 21, 26, 28, 34, 37]

    # files used for validation:
    validation_nums = [30, 32]

    # generate and save the training and validation set:
    overlap = 0.6
    x_train, y_train = generate_dataset(data_path, training_nums, overlap)
    x_val, y_val = generate_dataset(data_path, validation_nums, overlap)

    save_dataset(path.join(output_path,'vaihingen_train.hdf5'), x_train, y_train)
    save_dataset(path.join(output_path,'vaihingen_val.hdf5'), x_val, y_val)


if __name__ == '__main__':

    
    if len(sys.argv)<2:
      print ''
      print '***********************************'
      print 'Two parameters need to be specified:'
      print '1. Location of the input image patches (e.g., /homea/hpclab/train001/data/vaihingen/ )'
      print '2. Location to write training and validation sets (e.g., /homea/hpclab/train002/semseg/vaihingen/ )'
      print '***********************************'

      sys.exit()

    main(argv)







