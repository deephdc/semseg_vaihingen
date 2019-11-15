# -*- coding: utf-8 -*-
"""
Model description
"""
import os
import json
import yaml
import argparse
import zipfile
import pkg_resources
import subprocess
from keras import backend

import flask
from werkzeug.exceptions import BadRequest

# import project's config.py
import semseg_vaihingen.config as cfg
import semseg_vaihingen.models.train_resnet50_fcn as train_resnet50
import semseg_vaihingen.models.evaluate_network as predict_resnet50
import semseg_vaihingen.models.create_resfiles as resfiles 

#from datetime import datetime

def byte2str(str_in):
    '''
    Simple function to decode a byte string (str_in).
    In case of a normal string, return is unchanged
    '''
    try:
        str_in = str_in.decode()
    except (UnicodeDecodeError, AttributeError):
        pass
    
    return str_in 
    
def rclone_copy(src_path, dest_path, cmd='copy',):
    '''
    Wrapper around rclone to copy files
    :param src_path: path of what to copy. in the case of "copyurl" path at the remote
    :param dest_path: path where to copy
    :param cmd: how to copy, "copy" or "copyurl"
    :return: output message and a possible error
    '''

    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_path])
    elif cmd == 'copyurl':
        src_path = '/' + src_path.lstrip('/')
        src_dir, src_file = os.path.split(src_path)
        remote_link = cfg.MODEL_REMOTE_PUBLIC + src_dir + '&files=' + src_file
        print("[INFO] Trying to download {} from {}".format(src_file,
                                                            remote_link))
        command = (['rclone', 'copyurl', remote_link, dest_path])
    else:
        message = "[ERROR] Wrong 'cmd' value! Allowed 'copy', 'copyurl', received: " + cmd
        raise Exception(message)

    try:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e
        
    output = byte2str(output)
    error = byte2str(error)
    
    return output, error

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
            if line.startswith(par+":"):
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
            #catch_data_error(f.filename) 

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
            model_retrieve = yaml.safe_load(arg.model_retrieve)
            if not os.path.exists(cfg.MODEL_PATH) or model_retrieve:
                model_dir, model_file = os.path.split(cfg.MODEL_PATH)
                model_file_zip = model_file + '.zip'
                remote_src_path = os.path.join('models', model_file_zip)
                store_zip_path = os.path.join(model_dir, model_file_zip)
                print("[INFO] File {} will be retrieved from the remote.".format(store_zip_path))
                output, error = rclone_copy(src_path=remote_src_path,
                                            dest_path=store_zip_path,
                                            cmd='copyurl')
                if error:
                    message = "[ERROR] File was not properly copied. rclone returned: "
                    message = message + error
                    raise Exception(message)
                
                # if .zip is present locally, de-archive it
                if os.path.exists(store_zip_path):
                    print("[INFO] {} was downloaded. Unzipping...".format(store_zip_path))
                    data_zip = zipfile.ZipFile(store_zip_path, 'r')
                    data_zip.extractall(model_dir)
                    data_zip.close()
                    # remove downloaded zip-file
                    if os.path.exists(model_dir):
                        os.remove(store_zip_path)


            # Error catch: wrong image format
            filename, ext = os.path.splitext(f.name)
            ext = ext.lower()
            print("[DEBUG] filename: {}, ext: {}".format(filename, ext))
            data_type = 'any'
            if ext == '.hdf5' and "vaihingen_" in filename:
                prediction = predict_resnet50.predict_complete_image(f.name, 
                                                                     model)
                data_type = 'vaihingen'
            elif ( ext == '.jpeg' or ext == '.jpg' or ext == '.png' 
                   or ext == '.tif' or ext == '.tiff' ):
                prediction = predict_resnet50.predict_complete_image_jpg(f.name, 
                                                                         model)
            else:
                raise BadRequest(""" [ERROR] Image format error: \
                    Only '.hdf5', '.jpg', '.png', or 'tif' files are allowed. """)

            prediction_results["prediction"].update(prediction)

        except Exception as e:
            raise e
        finally:
            os.remove(f.name)

    # Build result file and stream it back
    result_pdf = resfiles.create_pdf(prediction_results["prediction"],
                                     data_type=data_type)

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
    
    # check if vaihingen_train.hdf5 and vaihingen_val.hdf5 exist locally,
    # if not -> download them from the REMOTE_STORAGE
    training_data = os.path.join(cfg.DATA_DIR, cfg.TRAINING_DATA)
    validation_data = os.path.join(cfg.DATA_DIR, cfg.VALIDATION_DATA)
    remote_data_storage = os.path.join(cfg.REMOTE_STORAGE, 'data')
    if not (os.path.exists(training_data) or os.path.exists(validation_data)):
        print("[INFO] Either %s or %s NOT found locally, download them from %s" % 
              (training_data, validation_data, remote_data_storage))
        output, error = rclone_copy(remote_data_storage, cfg.DATA_DIR)
        if error:
            message = "[ERROR] training data not copied. rclone returned: " + error
            raise Exception(message)

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

    # REMOTE_MODELS_UPLOAD is defined in config.py #vk
    upload_back = yaml.safe_load(train_args.upload_back)
    if(upload_back and os.path.exists(cfg.MODEL_PATH)):
        # zip the trained model, aka savedmodel:
        # adapted from https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory-in-python
        model_dir, model_file = os.path.split(cfg.MODEL_PATH)
        # full path to the zip file
        model_zip_path = os.path.join(cfg.MODEL_PATH, model_file + '.zip')
        # cd to the directory with the trained model
        os.chdir(model_dir)
        graph_zip = zipfile.ZipFile(model_zip_path, 'w', zipfile.ZIP_DEFLATED)
        graph_zip.write(model_file)
        graph_zip.close()        
        
        output, error = rclone_copy(model_zip_path, cfg.REMOTE_MODELS_UPLOAD)
        if error:
            print("[ERROR] rclone returned: {}".format(error))
    else:
        print("[ERROR] Created weights file, %s, was NOT uploaded!" % cfg.MODEL_PATH)

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


# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.predict_args

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        #print(val['default'], type(val['default']))

    return predict_args


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
