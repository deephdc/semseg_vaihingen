# imports
import semseg_vaihingen.config as cfg
from . import model_generator
from . import data_io as dio

import numpy as np
from sklearn import metrics
import keras
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os, re
import argparse

# label list
glob_label_list = np.array(['Impervious surfaces', 'Building',
                            'Low vegetation', 'Tree',
                            'Car', 'Clutter/Background'])

# dictionary with mapping {label - color}:
glob_label_color_dict = {'Impervious surfaces':'gray',
                         'Building':           'red',
                         'Low vegetation':     'lightgreen',
                         'Tree':               'green',
                         'Car':                'purple',
                         'Clutter/Background': 'black' }

# calculate the networks prediction at a given window of the image:
def net_predict(data, model, s, k, l):
    x = np.reshape(data[k:k + s, l:l + s, :], (1, s, s, 3))
    y = np.reshape(model.predict(x), (s, s, 6))
    return from_categorical(y) + 1


# inverse function to to_categorical:
def from_categorical(categorical_tensor):
    return np.argmax(categorical_tensor, axis=2)


# function to generate a plot of the ground truth or network prediction:
def create_colormap(label_matrix, title, labels=glob_label_list, 
                    colormap=True, legend=False):

    fig, ax1 = plt.subplots()
    if legend:
        #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # Fake plots to create legend
        for label in labels:
            ax1.plot(0, 0, "o", c=glob_label_color_dict[label], label=label)
        plt.subplots_adjust(right=0.75)
        ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    
    if colormap:
        # create custom colormap:
        colors = [ glob_label_color_dict[label] for label in labels ]
        label_cmap = ListedColormap(colors)
    
        # generate and show the map
        ax1.imshow(label_matrix, cmap=label_cmap)
    else:
        print(("[DEBUG] label_matrix.shape={}".format(label_matrix.shape)))
        ax1.imshow(label_matrix)

    plt.title(title)
    #plt.show()
    
    plot_file = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join(cfg.BASE_DIR, 'data', plot_file), 
                dpi=225, bbox_inches='tight')
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
    overall = np.sum(confusion, axis=1)

    print('')
    print("[INFO] Results:")
    print('Labelwise accuracy: ')
    print(('{:20s} \t {:>10s} \t {:>10s}'.format("Labels", "pixels", "accuracy")))
    print(("-".rjust(50,"-")))
    for i, label in enumerate(glob_label_list):
        print(('{:20s} \t {:10d} \t {:10.4f}%'.format(label, 
                                                     overall[i], 
                                                     100.*label_accuracy[i])))
    print('')


# function to apply a trained network to a whole image:
def predict_complete_image(patch_path, network_weight_file, 
                           convert_gray=False):
    image_number = re.search('_(.*).hdf5', patch_path).group(1)
    print(('[INFO] Load image number {} ... '.format(image_number)))
    data, ground_truth = dio.load_vaihingen_image(patch_path, image_number,
                                                  convert_gray=convert_gray)
    print(('[INFO] Image size: (%d x %d)' % (data.shape[0], data.shape[1])))

    # plot the input:
    create_colormap(data, title='Input image patch', colormap=False)

    num_labels_in_ground_truth = int(np.max(ground_truth))
    label_indecies = np.arange(num_labels_in_ground_truth).tolist()
    labels_in_ground_truth = glob_label_list[label_indecies]
    print(("[DEBUG] label indecies: {}".format(label_indecies)))
    print(("[DEBUG] num_labels_ground_truth={}, labels={}".format(
                                                   num_labels_in_ground_truth,
                                                   labels_in_ground_truth)))

    # create a colormap of the ground truth:
    create_colormap(ground_truth, title='Groundtruth',
                    labels=labels_in_ground_truth,
                    colormap=True, legend=True)

    print(('[INFO] Load a trained FCN from {} ...'.format(network_weight_file)))
    model = model_generator.generate_resnet50_fcn(use_pretraining=False)
    model.load_weights(network_weight_file)

    # preprocess (center, normalize) the input, using Keras' build in routine:
    data = keras.applications.resnet50.preprocess_input(data.astype(np.float32), mode='tf')

    # define image size and network input/output size:
    im_h = data.shape[0]
    im_w = data.shape[1]
    s = cfg.PATCH_SIZE

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
    print(('[INFO] Overall accuracy: %0.2f'% overall_acc))
    results["overall_accuracy"] = '%0.2f' % float(overall_acc)

    # calculate the confusion matrix:
    confusion, label_accuracy = analyze_result(ground_truth, 
                                               prediction, 
                                               cfg.NUM_LABELS)
    print_labelwise_accuracy(confusion, label_accuracy)
    print('[INFO] Confusion matrix: ')
    print(confusion)

    # store the % of correct predicted pixels per label in a dict
    results["label_accuracy"] = {}
    i_label = 0
    for label in glob_label_list:
        results["label_accuracy"][label] = "{}%".format(100.*
                                                       label_accuracy[i_label])
        i_label += 1

    num_labels_in_prediction = int(np.max(prediction))
    label_indecies = np.arange(num_labels_in_prediction).tolist()
    labels_in_prediction = glob_label_list[label_indecies]
    # create a colormap showing the networks predictions:  
    create_colormap(prediction, title='Classification map', 
                    labels = labels_in_prediction, legend=True)
 
    return results

# function to apply a trained network to a whole image:
def predict_complete_image_jpg(patch_path, network_weight_file,
                               convert_gray=False):

    data = dio.load_image_jpg(patch_path, convert_gray=convert_gray)
    print(('[INFO] Image size: (%d x %d)' % (data.shape[0], data.shape[1])))
    total_pixels = data.shape[0]*data.shape[1]

    # plot the input:
    create_colormap(data, title='Input image patch', colormap=False)

    print(('[INFO] Load a trained FCN from {} ...'.format(network_weight_file)))
    model = model_generator.generate_resnet50_fcn(use_pretraining=False)
    model.load_weights(network_weight_file)

    # preprocess (center, normalize) the input, using Keras' build in routine:
    data = keras.applications.resnet50.preprocess_input(data.astype(np.float32), mode='tf')

    # define image size and network input/output size:
    im_h = data.shape[0]
    im_w = data.shape[1]
    s = cfg.PATCH_SIZE

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

    num_labels_in_prediction = int(np.max(prediction))
    label_indecies = np.arange(num_labels_in_prediction).tolist()
    labels_in_prediction = glob_label_list[label_indecies]
    # create a colormap showing the networks predictions:  
    create_colormap(prediction, title='Classification map', 
                    labels = labels_in_prediction, legend=True)

    results = { "total_pixels" : int(total_pixels),
                "label_pixels" : {},
                "label_pixels_fraction": {}
              }

    print(("[DEBUG] unqiue values in prediction: {}".format(np.unique(prediction))))
    
    i_label = 0
    for label in glob_label_list:
        label_sum = (prediction == float(i_label + 1.)).sum()
        results["label_pixels"][label] = int(label_sum)
        results["label_pixels_fraction"][label] = float(np.round(
                                                label_sum/float(total_pixels),
                                                5))
        i_label += 1

    print("[INFO] Results:")
    print(('{:20s} \t {:>12s} \t {:>8s}'.format("Labels", "pixels", "fraction")))
    print(("-".rjust(48,"-")))
    for label, value in list(results["label_pixels"].items()):
        print(('{:20s}: \t {:12d} \t {:8f}'.format(label, value, 
                                      results["label_pixels_fraction"][label])))
    print(('{:20s}: \t {:12d}'.format("Total pixels", results["total_pixels"])))
 
    return results

def main():
    res = predict_complete_image(args.patch_path, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--patch_path', type=str,
                        help='Location of of the patch to test \
                        (e.g., /srv/semseg_vaihingen/data/raw/vaihingen_15.hdf5 )')
    parser.add_argument('--model', type=str,
                        help='Location of the trained network model \
                        (e.g., /srv/semseg_vaihingen/models/resnet50_fcn_weights.hdf5)')

    args = parser.parse_args()

    main()
