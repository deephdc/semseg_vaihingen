# imports:
import model_generator
import data_io as dio

import numpy as np
from sklearn import metrics
import keras
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sys
from sys import argv


# calculate the networks prediction at a given window of the image:
def net_predict(data, model, s, k, l):
    x = np.reshape(data[k:k + s, l:l + s, :], (1, s, s, 3))
    y = np.reshape(model.predict(x), (s, s, 6))
    return from_categorical(y) + 1


# inverse function to to_categorical:
def from_categorical(categorical_tensor):
    return np.argmax(categorical_tensor, axis=2)


# function to generate a plot of the ground truth or network prediction:
def create_colormap(label_matrix,title):
    # dictionary with mapping {label - color}:
    lc = {'surfaces':    (0.592, 0.647, 0.647),#'gray',
          'building':    (0.949, 0.109, 0.109),#'red',
          'vegetation':  (0.619, 0.878, 0.627),#'lightgreen',
          'tree':        (0.223, 0.498, 0.231),#'green',
          'car':         (0.564, 0.298, 0.482),#'purple',
          'clutter':     (0,0,0)}#'black'

    # create custom colormap:
    label_cmap = ListedColormap([lc['surfaces'],
                                 lc['building'],
                                 lc['vegetation'],
                                 lc['tree'],
                                 lc['car'],
                                 lc['clutter']])
    
    # generate and show the map
    plt.imshow(label_matrix, cmap=label_cmap)
    plt.title(title)
    plt.show()

    #plt.savefig('/gpfs/homeb/pcp0/pcp0096/semseg/out/ground_truth.png')


# function to generate a plot of the ground truth or network prediction if there are only 5 classes in the image
def create_colormap_no_clutter(label_matrix,title):
    # dictionary with mapping {label - color}:
    lc = {'surfaces':   'gray',
          'building':   'red',
          'vegetation': 'lightgreen',
          'tree':       'green',
          'car':        'purple'}

    # create custom colormap:
    label_cmap = ListedColormap([lc['surfaces'],
                                 lc['building'],
                                 lc['vegetation'],
                                 lc['tree'],
                                 lc['car']])

    # generate and show the map
    plt.imshow(label_matrix, cmap=label_cmap)
    plt.show()

    #plt.savefig('/gpfs/homeb/pcp0/pcp0096/semseg/out/classification_map.png')


# function to generate a plot of the wrong classified pixels
def create_errormap(error_matrix,title):
    # create custom colormap:
    error_cmap = ListedColormap(['black', 'white'])

    # generate and show the map
    plt.imshow(error_matrix, cmap=error_cmap)
    plt.title(title)
    plt.show()

    #plt.savefig('/gpfs/homeb/pcp0/pcp0096/semseg/out/error_map.png')


# calculate the confusion matrix and the class accuracy:
def analyze_result(ground_truth, prediction, num_labels):
    # reshape to one dimensional arrays:
    ground_truth = np.ravel(ground_truth)
    prediction = np.ravel(prediction)

    # calculate the confusion matrix:
    confusion = metrics.confusion_matrix(ground_truth, prediction, np.arange(num_labels)+1)

    # labelwise accuracy = correctly classified pixels of this label / pixels of this label
    label_accuracy = np.diag(confusion.astype(float)) / np.sum(confusion.astype(float), axis=1)


    return confusion, np.round(label_accuracy, 3)


# print the labelwise accuracy
def print_labelwise_accuracy(confusion, label_accuracy):
    labels = ['Impervious surfaces    ',
              'Building               ',
              'Low vegetation         ',
              'Tree                   ',
              'Car                    ',
              'Clutter/background     ']
    overall = np.sum(confusion, axis=1)
    print('')
    print('Labelwise accuracy: ')
    print('Labels             \t\t pixels     \t accuracy')
    for i, label in enumerate(labels):
        print('{} \t {}\t\t {}%'.format(label, overall[i], 100*label_accuracy[i]))
    print('')




# function to apply a trained network to a whole image:
def predict_complete_image(image_number, network_weight_file):

    print('--> Load image number {} ... '.format(image_number))
    image_name = '/homea/hpclab/train001/data/vaihingen/vaihingen_' + str(image_number) + '.hdf5'
    data, ground_truth = dio.load_vaihingen_image(image_name, image_number)
    print('Image size: (', data.shape[0], 'x', data.shape[1], ')')

    num_labels = 6
    num_labels_in_ground_truth = np.max(ground_truth)

    # plot the input:
    create_colormap(data,'Image patch')



    # create a colormap of the ground truth:
    if num_labels_in_ground_truth == num_labels:
        create_colormap(ground_truth,'Groundtruth')
    else:
        create_colormap_no_clutter(ground_truth,'Groundtruth')

    print('--> Load a trained FCN from {} ...'.format(network_weight_file))
    model = model_generator.generate_resnet50_fcn(use_pretraining=False)
    model.load_weights(network_weight_file)

    # preprocess (center, normalize) the input, using Keras' build in routine:
    data = keras.applications.resnet50.preprocess_input(data.astype(np.float32), mode='tf')

    # define image size and network input/output size:
    im_h = data.shape[0]
    im_w = data.shape[1]
    s = 256

    print('--> Apply network to image ... ')
    # iterate over the complete image:
    prediction = np.zeros((im_h, im_w))
    k = l = 0
    while k+s < im_h:
        while l+s < im_w:
            prediction[k:k+s, l:l+s] = net_predict(data, model, s, k, l)
            l += s
        # right border:
        l = im_w - s
        prediction[k:k + s, l:l + s] = net_predict(data, model, s, k, l)
        k += s
        l = 0
    # bottom border:
    k = im_h - s
    while l + s < im_w:
        prediction[k:k + s, l:l + s] = net_predict(data, model, s, k, l)
        l += s
    # right border:
    l = im_w - s
    prediction[k:k + s, l:l + s] = net_predict(data, model, s, k, l)

    # create a colormap showing the networks predictions:
    create_colormap(prediction,'Classification map')

    print('--> Calculate error map ... ')
    # create a map, showing which pixels were predicted wrongly:
    error_map = np.zeros((im_h, im_w))
    num_cor = 0
    for k in range(im_h):
        for l in range(im_w):
            if prediction[k, l] == ground_truth[k, l]:
                error_map[k, l] = 1
                num_cor += 1
    create_errormap(error_map,'Misclassified pixels map')

    print('--> Analyze the network accuracy ... ')

   
    print 'Overall accuracy: %0.2f'% np.divide(float(100*num_cor), float(im_w*im_h))
    #print('Overall accuracy: {}%'.format(np.round(100*num_cor/(im_w*im_h), 1)))

    # calculate the confusion matrix:
    confusion, label_accuracy = analyze_result(ground_truth, prediction, 6)

   

    print_labelwise_accuracy(confusion, label_accuracy)

    print('Confusion matrix: ')
    print(confusion)


def main(arguments):
    patch = arguments[1]
    model = arguments[2]
    predict_complete_image(patch, model)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print ''
        print '***********************************'
        print 'Two parameters need to be specified:'
        print '1. The number of the patch to test (e.g., 15 ,23)'
        print '2. The network model (e.g., /homea/hpclab/train002/semseg/models/resnet50_fcn_weights.hdf5)'
        print '***********************************'

        sys.exit()

    main(argv)
