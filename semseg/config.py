# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))


# Training arguments as a dict of dicts 
# with the following structure to feed the deepaas API parser:
# (see also get_train_args() )
# { 'arg1' : {'default': 1,       # default value
#             'help': '',         # can be an empty string
#             'required': False   # bool
#             },
#   'arg2' : {'default': 'value1',
#             'choices': ['value1', 'value2', 'value3'],
#             'help': 'multi-choice argument',
#             'required': False
#             },
#   'arg3' : {...
#             },
# ...
# }
train_args = { 'data_path': {'default': '/srv/semseg/data',
                        'help': 'Location of vaihingen_train.hdf5 and vaihingen_val.hdf5',
                        'required': False
                        },
                'model':    {'default': '/srv/semseg/models/resnet50_fcn_weights.hdf5',
                             'help': 'Location + name of the output model \
                             (e.g., /srv/semseg/models/resnet50_fcn_weights.hdf5)',
                             'required': False
                             },
                'no_augmentation': {'default': True,
                                    'choices': [True, False],
                                    'help': 'Skip augmentation',
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
                             }
}

#    parser.add_argument('--data_path', type=str,
#                        help='Location of vaihingen_train.hdf5 and vaihingen_val.hdf5 \
#                        (e.g., /homea/hpclab/train002/semseg/data/ )')
#    parser.add_argument('--model', type=str,
#                        help='Location + name of the output model \
#                        (e.g., /homea/hpclab/train002/semseg/models/resnet50_fcn_weights.hdf5)')
#    parser.add_argument('--n_epochs', type=int, default=20, 
#                        help='Number of epochs to train on')
#    parser.add_argument('--n_gpus', type=int, default=1,
#                        help='Number of GPUs to train on (one node only!)')
#    parser.add_argument('--no_augmentation', dest='augmentation', default=True,
#                        action='store_false', help='Skip augmentation')
#
#    parser.add_argument('--load_weights', dest='transfer_learning', default=False,
#                        action='store_true', 
#                        help='Use transfer learning and load pre-trained weights')
#    parser.add_argument('--log', type=str,
#                        help='Location + name of the csv log file')  