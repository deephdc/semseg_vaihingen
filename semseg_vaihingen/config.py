# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))

DATA_DIR = path.join(BASE_DIR,'data') # Location of model data and output files
MODEL_WEIGHTS_FILE = 'resnet50_fcn_weights.hdf5'
MODEL_DIR = path.join(BASE_DIR,'models') # Location + name of the output model
MODEL_REMOTE_PUBLIC = 'https://nc.deep-hybrid-datacloud.eu/s/eTqJexZ5PcBxXR6/download?path='
REMOTE_STORAGE = 'rshare:/deep-oc-apps/semseg_vaihingen'
REMOTE_MODELS_UPLOAD = path.join(REMOTE_STORAGE, 'models')
NUM_LABELS = 6  # max number of labels
PATCH_SIZE = 256
TRAINING_DATA = 'vaihingen_train.hdf5'
VALIDATION_DATA = 'vaihingen_val.hdf5'

train_args = { 'augmentation': {'default': False,
                                'choices': [False, True],
                                'help': 'Apply augmentation',
                                'required': False
                                },
               'transfer_learning': {'default': False,
                                      'choices': [False, True],
                                      'help': 'Use transfer learning and load pre-trained weights',
                                      'required': False
                                    },
               'n_gpus':   {'default': 1,
                            'help': 'Number of GPUs to train on (one node only!)',
                            'required': False
                           },
               'n_epochs': {'default': 20,
                            'help': 'Number of epochs to train on',
                            'required': False 
                           },
               'batch_size':  {'default': 16,
                               'help': 'Number of samples per batch',
                               'required': False
                              },
               'upload_back': {'default': False,
                               'choices': [False, True],
                               'help': 'Either upload a trained model back to the remote storage (True) or not (False, default)',
                               'required': False
                              },
               'model_weights_save':  {'default': MODEL_WEIGHTS_FILE,
                                       'help': 'Filename for the models weights',
                                       'required': False
                                       },
}

predict_args = {'model_weights_load':  {'default': MODEL_WEIGHTS_FILE,
                                        'help': 'Filename for the models weights (default: resnet50_fcn_weights.hdf5)',
                                        'required': False
                                        },
                'model_retrieve':   {'default': False,
                             'choices': [False, True],
                             'help': 'Force model update from the remote repository',
                             'required': False
                           },
               #'convert_grayscale':   {'default': True,
               #              'choices': [False, True],
               #              'help': 'Convert color image to grayscale before processing (default)',
               #              'required': False
               #            },
}
 
