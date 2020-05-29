# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""
import os
from os import path
from webargs import fields, validate


# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = path.join(IN_OUT_BASE_DIR, 'data') # Location of model data and output files
MODEL_DIR = path.join(IN_OUT_BASE_DIR, 'models') # Location + name of the output model

MODEL_WEIGHTS_FILE = 'resnet50_fcn_weights.hdf5'

MODEL_REMOTE_PUBLIC = 'https://nc.deep-hybrid-datacloud.eu/s/eTqJexZ5PcBxXR6/download?path='
REMOTE_STORAGE = 'rshare:/semseg_vaihingen'
REMOTE_MODELS_UPLOAD = path.join(REMOTE_STORAGE, 'models')
NUM_LABELS = 6  # max number of labels
PATCH_SIZE = 256
TRAINING_DATA = 'vaihingen_train.hdf5'
VALIDATION_DATA = 'vaihingen_val.hdf5'

train_args = { 'augmentation': fields.Str(
                    missing = False,
                    enum = [False, True],
                    description = 'Apply augmentation',
                    required = False
                ),
               'transfer_learning': fields.Str(
                    missing = False,
                    enum = [False, True],
                    description = 'Use transfer learning and load pre-trained weights',
                    required  =False
               ),
               'n_gpus': fields.Str(
                   missing = 1,
                   description = 'Number of GPUs to train on (one node only!)',
                   required = False
               ),
               'n_epochs': fields.Str(
                   missing = 20,
                   description = 'Number of epochs to train on',
                   required = False 
               ),
               'batch_size':  fields.Str(
                   missing = 16,
                   description = 'Number of samples per batch',
                   required = False
               ),
               'upload_back': fields.Str(
                  missing = False,
                  enum = [False, True],
                  description = 'Either upload a trained model back to the remote storage (True) or not (False, default)',
                  required = False
               ),
               'model_weights_save': fields.Str(
                   missing = MODEL_WEIGHTS_FILE,
                   description = 'Filename for the models weights',
                   required = False
               ),
}

predict_args = { "files": fields.Field(
                    description="Data file to perform inference on (vaihingen_#.hdf5 or any .tiff, .png, .jpg file).",
                    required = True,
                    type="file",
                    location="form"
                ),
    
                'model_weights_load': fields.Str(
                    missing= MODEL_WEIGHTS_FILE,
                    description = 'Filename for the models weights (default: resnet50_fcn_weights.hdf5)',
                    required= False
                ),
                'model_retrieve':fields.Str(
                    missing = False,
                    enum = [False, True],
                    description = 'Force model update from the remote repository',
                    required =  False
                ),
                
                "accept" : fields.Str(
                    require =False,
                    description ="Returns an image or a json with the box coordinates.",
                    missing ='application/pdf',
                    validate =validate.OneOf(['application/pdf', 'application/json'])),
                
               #'convert_grayscale':   fields.Str(
               #     missing = True,
               #     enum =  [False, True],
               #     description = 'Convert color image to grayscale before processing (default)',
               #     required = False
               #            },
}
 
