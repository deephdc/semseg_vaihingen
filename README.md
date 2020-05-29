2D semantic segmentation
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code%2FDEEP-OC-org%2Fsemseg_vaihingen%2Fmaster)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/semseg_vaihingen/job/master/)
----

2D semantic segmentation (Vaihingen dataset)

**Author:** G.Cavallaro (FZJ), M.Goetz (KIT), V.Kozlov (KIT), A.Grupp (KIT)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has
received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is an example application for [ISPRS 2D Semantic Labeling Contest ](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html):
    "2D semantic segmentation (Vaihingen dataset) that assigns labels to multiple object categories.",

You can find more information about it in the [DEEP Marketplace](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-semseg-vaihingen.html).

**Table of contents**
1. [Installing this module](#installing-this-module)
    1. [Local installation](#local-installation)
    2. [Docker installation](#docker-installation)
2. [Train the classifier](#train-the-classifier)
    1. [Data preprocessing](#data-preprocessing)
    2. [Train the classifier](#train-the-classifier)
3. [Predict](#predict)
4. [Acknowledgements](#acknowledgments)

## Installing this module

### Local installation

> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6. Further package requirements are described in the
> `requirements.txt` file.

To start using this framework clone the repo:

```bash
git clone https://github.com/deephdc/semseg_vaihingen
cd semseg_vaihingen
pip install -e .
```
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui and look for the methods belonging to the `semseg_vaihingen` module.

### Docker installation

We have also prepared a ready-to-use [Docker container](https://github.com/deephdc/DEEP-OC-semseg_vaihingen) to
run this module. To run it:

```bash
docker search deephdc
docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 deephdc/deep-oc-semseg_vaihingen
```

Now open http://0.0.0.0:5000/ui and look for the methods belonging to the `semseg_vaihingen` module.


## Train a the classifier

### Data preprocessing

The first step to train the neural network is to put the training file `vaihingen_train.hdf5` and the validation file `vaihingen_val.hdf5` into `./semseg_vaihingen/data`. More information about how to acquire the vaihingen dataset can be found [here](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html).

A script that converts raw data into hdf5 file is also provided.

### Train the classifier

Go to http://0.0.0.0:5000/ui and look for the ``TRAIN`` POST method. Click on 'Try it out', change whatever training args
you want and click 'Execute'. The training will be launched and you will be able to follow its status by executing the 
``TRAIN`` GET method which will also give a history of all trainings previously executed.

If the module has some sort of training monitoring configured (like Tensorboard) you will be able to follow it at 
http://0.0.0.0:6006.

After training you can check training statistics and check the logs where you will be able to find the standard output
during the training together with the confusion matrix after the training was finished.

## Predict

This module comes with a pretrained classifier so you won't have to train the classifier first before being able to use the testing methods.

Go to http://0.0.0.0:5000/ui and look for the `PREDICT` POST method. Click on 'Try it out', change whatever test args
you want and click 'Execute'. You must supply a `data` argument with a path pointing to a `vaihingen_#.hdf5` or any `.tiff, .png, .jpg` file. You can also choose either getting the results as a `json` response or as a downloadable `.pdf` file.

## Acknowledgments

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692. 

