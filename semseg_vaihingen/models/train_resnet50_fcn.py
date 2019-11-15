#imports:
import os
import sys
import time
import argparse
import keras
import augmentation
import model_generator
import numpy as np
import data_io as dio
import storeincsv as incsv
import semseg_vaihingen.config as cfg
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import preprocess_input
from collections import namedtuple
from datetime import datetime

import tensorflow as tf
from keras.utils import multi_gpu_model

ParamHeader = ['Timestamp', 'Script', 'val_acc', 'val_loss',
               'TotalTime', 'MeanPerEpoch', 'StDev']
ParamEntry = namedtuple('ParamEntry', ParamHeader)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_duration = 0.
        self.durations = []
        self.val_acc = 0.
        self.val_loss = 0.

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        duration_epoch = time.time() - self.epoch_time_start
        self.total_duration += duration_epoch
        self.durations.append(duration_epoch)
        self.val_acc = logs.get('val_acc')
        self.val_loss = logs.get('val_loss')

def train(data_path,
          output_model,
          augmentation_flag,
          transfer_learning_flag,
          n_gpus,
          n_epochs,
          batch_size):

    print('[INFO] Load data ... ')
    x_train, y_train = dio.load_data(os.path.join(data_path, cfg.TRAINING_DATA))
    print('[INFO] Training samples: {}'.format(x_train.shape))
    x_val, y_val = dio.load_data(os.path.join(data_path, cfg.VALIDATION_DATA))
    print('[INFO] Validation samples: {}'.format(x_val.shape))

    if augmentation_flag:
        print("[INFO] With augmentation:")
        print('.[INFO] Generate augmented images ...')
        x_train_aug, y_train_aug = augmentation.every_element_randomly_once(x_train, y_train)
        x_val_aug, y_val_aug = augmentation.every_element_randomly_once(x_val, y_val)
        #x_train_aug, y_train_aug = augmentation.every_element_five_augmentations(x_train, y_train)
        #x_val_aug, y_val_aug = augmentation.every_element_five_augmentations(x_val, y_val)

        print(x_train.dtype)
        print(x_train_aug.dtype)

        # put each array together with its augmented version:
        print('.[INFO] Concatenate data with original and augmented images')
        x_train = np.concatenate([x_train, x_train_aug])
        y_train = np.concatenate([y_train, y_train_aug])
        x_val = np.concatenate([x_val, x_val_aug])
        y_val = np.concatenate([y_val, y_val_aug])

        # shuffle the samples:
        print('.[INFO] Shuffle samples')
        x_train, y_train = augmentation.shuffle_4d_sample_wise(x_train, y_train)
        x_val, y_val = augmentation.shuffle_4d_sample_wise(x_val, y_val)


    print('[INFO] Preprocess the input data (normalization, centering) ... ')
    x_train = preprocess_input(x_train, mode='tf')
    x_val = preprocess_input(x_val, mode='tf')

    print('[INFO] Preprocess the labels ... ')
    y_train -= 1
    y_val -= 1
    y_train = to_categorical(y_train, cfg.NUM_LABELS)
    y_val = to_categorical(y_val, cfg.NUM_LABELS)

    print('[INFO] Load model ... ')
    if n_gpus > 1:
        with tf.device('/device:CPU:0'):
            resnet50_fcn_model_tmpl = model_generator.generate_resnet50_fcn(
                                      use_pretraining=transfer_learning_flag)

        resnet50_fcn_model_tmpl.summary()
        print("[INFO] Using {} GPUs".format(n_gpus))
        resnet50_fcn_model = multi_gpu_model(resnet50_fcn_model_tmpl, 
                                             gpus=n_gpus)
    else:
        resnet50_fcn_model = model_generator.generate_resnet50_fcn(
                             use_pretraining=transfer_learning_flag)

    resnet50_fcn_model.summary()

    print('[INFO] Compile model ... ')
    resnet50_fcn_model.compile(optimizer=keras.optimizers.Adam(),
                               loss=keras.losses.categorical_crossentropy,
                               metrics=['accuracy'])

    time_callback = TimeHistory()

    checkpointer = ModelCheckpoint(filepath=output_model, 
                                   verbose=1, save_best_only=True)
    resnet50_fcn_model.fit(x_train, y_train, 
                           batch_size=batch_size, 
                           epochs=n_epochs, 
                           validation_data=(x_val, y_val),
                           #steps_per_epoch=len(x_train)//batch_size,
                           callbacks=[checkpointer, time_callback],
                           verbose=1)

    print(time_callback.total_duration)

    # comment as we save 'best' weights via checkpointer
    #resnet50_fcn_model.save_weights(output_model)

    _mn = time_callback.total_duration / n_epochs

    mn = np.mean(time_callback.durations)
    sd = np.std(time_callback.durations, ddof=1)

    print("[INFO] Mean: ", _mn, mn)

    return ParamEntry(datetime.now(), os.path.basename(__file__),
                      time_callback.val_acc, time_callback.val_loss, 
                      time_callback.total_duration, mn, sd)


def main():
    
    debug = False
    
    param_entries = []
    param_entries.append(ParamHeader)
  
    if debug:
        print(args.data_path)
        print(args.model)
        print(args.augmentation)
        print(args.transfer_learning)
        print(args.n_epochs)
        print(args.batch_size)
    
    if args.augmentation:
        print(">> with augmentation")
    else:
        print(">> no augmentation")

    params = train(args.data_path, 
                   args.model, 
                   args.augmentation,
                   args.transfer_learning,
                   args.n_gpus,
                   args.n_epochs,
                   args.batch_size)

    param_entries.append(params)
    
    if args.log:
        incsv.store_data_in_csv(args.log, param_entries)    

    #sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str,
                        help='Location of vaihingen_train.hdf5 and vaihingen_val.hdf5 \
                        (e.g., /homea/hpclab/train002/semseg_vaihingen/data/ )')
    parser.add_argument('--model', type=str,
                        help='Location + name of the output model \
                        (e.g., /homea/hpclab/train002/semseg_vaihingen/models/resnet50_fcn_weights.hdf5)')
    parser.add_argument('--n_epochs', type=int, default=20, 
                        help='Number of epochs to train on')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of samples per batch')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to train on (one node only!)')
    parser.add_argument('--augmentation', dest='augmentation', default=False,
                        action='store_true', help='Apply augmentation')
    parser.add_argument('--load_weights', dest='transfer_learning', default=False,
                        action='store_true', 
                        help='Use transfer learning and load pre-trained weights')
    parser.add_argument('--log', type=str,
                        help='Location + name of the csv log file')                        
                        
    args = parser.parse_args()
    
#    if len(sys.argv)<4:
#      print ''
#      print '***********************************'
#      print 'Four paremeters need to be specified:'
#      print '1. Location of vaihingen_train.hdf5 and vaihingen_val.hdf5 (e.g., /homea/hpclab/train002/semseg/data/ )'
#      print '2. Location + name of the output model (e.g., /homea/hpclab/train002/semseg/models/resnet50_fcn_weights.hdf5)'
#      print '3. Augmentation: True or False'
#      print '4. Transfer learning (load weights trained on ImageNet): True or False'
#      print '***********************************'
#
#      sys.exit()

    main()
