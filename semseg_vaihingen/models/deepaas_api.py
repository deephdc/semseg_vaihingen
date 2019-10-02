# -*- coding: utf-8 -*-
"""
Model description
"""
import os
import json
import yaml
import argparse
import pkg_resources
from keras import backend

import flask
from werkzeug.exceptions import BadRequest

# import project's config.py
import semseg_vaihingen.config as cfg
import semseg_vaihingen.models.train_resnet50_fcn as train_resnet50
import semseg_vaihingen.models.evaluate_network as predict_resnet50
import semseg_vaihingen.models.create_resfiles as resfiles 

#from datetime import datetime

def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
        'Train-Args': cfg.train_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def catch_data_error(data):
    # Error catch: wrong image format
    extension = data.split('.')[-1]
    if extension != 'hdf5':
        raise BadRequest(""" Image format error:
        Only '.hdf5' files allowed. """)


def predict_file(path):
    """
    Function to make prediction on a local file
    """

    message = 'Not implemented in the model (predict_file)'
    return message


def predict_data(*args, **kwargs):
    """
    Function to make prediction on an uploaded image file
    """
    model = cfg.MODEL_PATH 
    prediction_results = { "status" : "ok",
                           "prediction": {} 
                         }
    
    imgs = []
    filenames = [] 
    
    # Check and store data
    for arg in args:
        files = arg['files']
        if not isinstance(files, list):
            files = [files]
        for f in files:
            imgs.append(f)
            catch_data_error(f.filename) 

    for image in imgs:
        image_name = image.filename
        f = open("/tmp/%s" % image_name, "w+")
        image.save(f.name)
        f.close
        filenames.append(f.name)
        print("Stored file (temporarily) at: {} \t Size: {}".format(f.name,
        os.path.getsize(f.name)))

        prediction_results["prediction"].update( {"file_name" : image_name} ) 
        # Perform prediction
        try: 
            # Clear possible pre-existing sessions. important!
            backend.clear_session()
            prediction = predict_resnet50.predict_complete_image(f.name, model)
            prediction_results["prediction"].update(prediction)

        except Exception as e:
            raise e
        finally:
            os.remove(f.name)

    # Build result file and stream it back
    result_image = resfiles.merge_images()
    result_pdf = resfiles.create_pdf(result_image,prediction_results["prediction"])

    return flask.send_file(filename_or_fp=result_pdf,
                           as_attachment=True,
                           attachment_filename=os.path.basename(result_pdf))

#    return prediction_results 


def predict_url(*args):
    """
    Function to make prediction on a URL
    """

    message = 'Not implemented in the model (predict_url)'
    return message


###
# Comment the following three lines if you open training for everybody, 
# i.e. do *not* need any authorization at all
###
from flaat import Flaat
flaat = Flaat()

@flaat.login_required()
def train(train_args):
    """
    Train network
    train_args : dict
        Json dict with the user's configuration parameters.
        Can be loaded with json.loads() or with yaml.safe_load()    
    """

    run_results = { "status": "ok",
                    "train_args": {},
                    "training": {},
                  }

    run_results["train_args"] = train_args

    # Clear possible pre-existing sessions. important!
    backend.clear_session()

    if (yaml.safe_load(train_args.augmentation)):
        params = train_resnet50.train_with_augmentation(
                                      cfg.DATA_DIR,
                                      cfg.MODEL_PATH,
                                      yaml.safe_load(train_args.augmentation),
                                      yaml.safe_load(train_args.transfer_learning),
                                      yaml.safe_load(train_args.n_gpus),
                                      yaml.safe_load(train_args.n_epochs),
                                      yaml.safe_load(train_args.batch_size))
    else:
        params = train_resnet50.train(cfg.DATA_DIR,
                                      cfg.MODEL_PATH,
                                      yaml.safe_load(train_args.augmentation),
                                      yaml.safe_load(train_args.transfer_learning),
                                      yaml.safe_load(train_args.n_gpus),
                                      yaml.safe_load(train_args.n_epochs),
                                      yaml.safe_load(train_args.batch_size))
    
    run_results["training"] = yaml.safe_load(json.dumps(params._asdict(), 
                                                        default=str))

    print("Run results: " + str(run_results))
    return run_results


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]

    return train_args


# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'train':
        train(args)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),
                            help=val['help'])

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    args = parser.parse_args()

    main()