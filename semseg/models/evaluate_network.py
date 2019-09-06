# imports
import semseg.config as cfg
import model_generator
import data_io as dio

import numpy as np
from sklearn import metrics
import keras
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os, re
import argparse


# calculate the networks prediction at a given window of the image:
def net_predict(data, model, s, k, l):
    x = np.reshape(data[k:k + s, l:l + s, :], (1, s, s, 3))
    y = np.reshape(model.predict(x), (s, s, 6))
    return from_categorical(y) + 1


# inverse function to to_categorical:
def from_categorical(categorical_tensor):
    return np.argmax(categorical_tensor, axis=2)


# function to generate a plot of the ground truth or network prediction:
def create_colormap(label_matrix,title, legend=False):
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

    if legend:
        # Fake plots to create legend
        for label in lc:
            plt.plot(0, 0, "o", c=lc[label], label=label)
        plt.subplots_adjust(right=0.75)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    # generate and show the map
    plt.imshow(label_matrix, cmap=label_cmap)
    plt.title(title)
    #plt.show()
    
    plot_file = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join(cfg.BASE_DIR, 'data', plot_file))
    plt.clf()


# function to generate a plot of the ground truth or network prediction if there are only 5 classes in the image
def create_colormap_no_clutter(label_matrix,title, legend=False):
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

    if legend:
        # Fake plots to create legend
        for label in lc:
            plt.plot(0, 0, "o", c=lc[label], label=label)
        plt.subplots_adjust(right=0.75)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    # generate and show the map
    plt.imshow(label_matrix, cmap=label_cmap)
    plt.title(title)    
    #plt.show()

    plot_file = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join(cfg.BASE_DIR, 'data', plot_file))
    plt.clf()


# function to generate a plot of the wrong classified pixels
def create_errormap(error_matrix,title):
    # create custom colormap:
    error_cmap = ListedColormap(['black', 'white'])

    # generate and show the map
    plt.imshow(error_matrix, cmap=error_cmap)
    plt.title(title)
    #plt.show()

    plt.savefig(os.path.join(cfg.BASE_DIR, 'data', 'Error_map.png'))


# calculate the confusion matrix and the class accuracy:
def analyze_result(ground_truth, prediction, num_labels):
    # reshape to one dimensional arrays:
    ground_truth = np.ravel(ground_truth)
    prediction = np.ravel(prediction)

    # calculate the confusion matrix:
    confusion = metrics.confusion_matrix(ground_truth, prediction, np.arange(num_labels)+1)

    # labelwise accuracy = correctly classified pixels of this label / pixels of this label
    true_pos = np.diag(confusion.astype(float))
    pred_pos = np.sum(confusion.astype(float), axis=1)
    label_accuracy = np.divide(true_pos, pred_pos, out=np.ones_like(true_pos),
            where=pred_pos!=0.0)

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
def predict_complete_image(patch_path, network_weight_file):
    image_number = re.search('_(.*).hdf5', patch_path).group(1)
    print('[INFO] Load image number {} ... '.format(image_number))
    data, ground_truth = dio.load_vaihingen_image(patch_path, image_number)
    print('[INFO] Image size: (%d x %d)' % (data.shape[0], data.shape[1]))

    num_labels = 6
    num_labels_in_ground_truth = np.max(ground_truth)

    # plot the input:
    create_colormap(data,'Input image patch')

    # create a colormap of the ground truth:
    if num_labels_in_ground_truth == num_labels:
        create_colormap(ground_truth,'Groundtruth', legend=True)
    else:
        create_colormap_no_clutter(ground_truth,'Groundtruth', legend=True)

    print('[INFO] Load a trained FCN from {} ...'.format(network_weight_file))
    model = model_generator.generate_resnet50_fcn(use_pretraining=False)
    model.load_weights(network_weight_file)

    # preprocess (center, normalize) the input, using Keras' build in routine:
    data = keras.applications.resnet50.preprocess_input(data.astype(np.float32), mode='tf')

    # define image size and network input/output size:
    im_h = data.shape[0]
    im_w = data.shape[1]
    s = 256

    print('[INFO] Apply network to image ... ')
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
    create_colormap(prediction,'Classification map', legend=True)

    print('[INFO] Calculate error map ... ')
    # create a map, showing which pixels were predicted wrongly:
    error_map = np.zeros((im_h, im_w))
    num_cor = 0
    for k in range(im_h):
        for l in range(im_w):
            if prediction[k, l] == ground_truth[k, l]:
                error_map[k, l] = 1
                num_cor += 1
    create_errormap(error_map,'Misclassified pixels map')


    print('[INFO] Analyze the network accuracy ... ')
    results = {}
    
    #print('Overall accuracy: {}%'.format(np.round(100*num_cor/(im_w*im_h), 1)))
    overall_acc = np.divide(float(100*num_cor), float(im_w*im_h))
    print('[INFO] Overall accuracy: %0.2f'% overall_acc)
    results["overall_accuracy"] = '%0.2f' % overall_acc

    # calculate the confusion matrix:
    confusion, label_accuracy = analyze_result(ground_truth, prediction, 6)
    print_labelwise_accuracy(confusion, label_accuracy)

    # store the % of correct predicted pixels per label in a dict
    results["label_accuracy"] = {}
    labels = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car ', 'Clutter/background']
    for i, label in enumerate(labels):
        results["label_accuracy"][label] = "{}%".format(100*label_accuracy[i])

    print('[INFO] Confusion matrix: ')
    print(confusion)

    return results


def main():
    res = predict_complete_image(args.patch_path, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--patch_path', type=str,
                        help='Location of of the patch to test \
                        (e.g., /srv/semseg/data/raw/vaihingen_15.hdf5 )')
    parser.add_argument('--model', type=str,
                        help='Location of the trained network model \
                        (e.g., /srv/semseg/models/resnet50_fcn_weights.hdf5)')

    args = parser.parse_args()

    main()
