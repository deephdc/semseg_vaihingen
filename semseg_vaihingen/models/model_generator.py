# imports:
from resnet50_edit import ResNet50
from keras.models import Model
from keras.layers import Input, Convolution2D, Lambda, Add, Reshape, Activation
import tensorflow as tf


# there is no layer for bilinear resizing in Keras, so we define this function and use it as a Lambda function
def resize_bilinear(images):
    rows = 256
    cols = 256
    return tf.image.resize_bilinear(images, [rows, cols])


# the function to generate a FCN version of the ResNet50 model:
def generate_resnet50_fcn(use_pretraining):
    num_labels = 6
    input_dim_row = 256
    input_dim_col = 256
    input_shape = (input_dim_row, input_dim_col, 3)
    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None
    standard_model = ResNet50(include_top=False, weights=weights, input_tensor=input_tensor)

    # get the activations after different network parts by name:
    x32 = standard_model.get_layer('act3d').output
    x16 = standard_model.get_layer('act4f').output
    x8 = standard_model.get_layer('act5c').output

    # apply 1x1 convolution to compress the depth of the output tensors to the number of classes:
    c32 = Convolution2D(filters=num_labels, kernel_size=(1, 1), name='conv_labels_32')(x32)
    c16 = Convolution2D(filters=num_labels, kernel_size=(1, 1), name='conv_labels_16')(x16)
    c8 = Convolution2D(filters=num_labels, kernel_size=(1, 1), name='conv_labels_8')(x8)

    # resize the spatial dimensions to fit the spatial input size:
    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    # sum up the activations of different stages to get information of different solution
    m = Add(name='merge_labels')([r32, r16, r8])

    # apply a softmax activation function to get the probability of each class for each pixel
    x = Reshape((input_dim_row * input_dim_col, num_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((input_dim_row, input_dim_col, num_labels))(x)

    # return the FCN version of the ResNet50 model:
    return Model(inputs=input_tensor, outputs=x)


# main method for printing the network structure:
def main():
    model = generate_resnet50_fcn(use_pretraining=True)
    model.summary()


if __name__ == '__main__':
    main()
