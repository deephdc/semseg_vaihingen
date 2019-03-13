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
train_args = { 'arg1': {'default': 1,
                        'help': '',
                        'required': False
                        },
}
