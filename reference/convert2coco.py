import os
from pathlib import Path
import json

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

import pandas as pd

from engine import train_one_epoch, evaluate
from utils import collate_fn, xView3BaselineModel
from train import create_datasets


# <h1 style="color:red">Disclaimer</h1>
# <p>This notebook is created using the "tiny" set of the xView3 dataset. This is a subsample of the data for ease-of-experimentation. You still need to download the label and shoreline files in their entirety.<p>

# In[13]:


# Run this cell if you have downloaded the "tiny" data partition; it will reformat that directory to
# be compatible with this notebook
# Specify directory within which tiny data partition has been downloaded and each folder unzipped

tiny_data_path = Path('/home/jovyan/xview3-tiny/')
# Creating directories
train_path = tiny_data_path / 'train'
val_path = tiny_data_path / 'validation'
if 0:
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Moving files
    for dr in os.listdir(tiny_data_path):
        if dr.endswith('t'):
            os.rename(f'{tiny_data_path}/{dr}', f'{tiny_data_path}/train/{dr}')
        elif dr.endswith('v'):
            os.rename(f'{tiny_data_path}/{dr}', f'{tiny_data_path}/validation/{dr}')


# ## Loading data
# xView3 data is made available as a set of GeoTIFFs per Scene ID. Along with [cross (VH) and co (VV) polarization channels](https://earth.esa.int/documents/10174/3166029/Vilnius_radar_general.pdf), we also include [bathymetry](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) and various wind layers.
# 
# <img src="figures/example_data.png" alt="xView 3 sample data" style="width:75%" />
# 
# Since these scenes are very large, the preprocessing code creates ML-ready chips and puts them into easy-to-use [PyTorch datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

# In[14]:


# Path to directory containing two directories 'train' and 'validation',
# each of which holds image folders downloaded from xView3 website.
# Each of these folders will have a uuid as its name (e.g. '590dd08f71056cacv')
# and will contain several .tif files.  To get a visualization consistent with what 
# we show here, make sure that scene '590dd08f71056cacv' is in the 'validation; directory;
# We recommend using ~ 5 scenes for training and ~2 scenes for validation as an initial proof-of-concept.  
# Note that the full xView3 training dataset contains several hundred scenes!
image_folder = tiny_data_path #'/home/jovyan/tiny/'

# Path to ground truth label files; should contain train.csv and validation.csv
label_file_root = tiny_data_path #'/home/jovyan/xview3'

# Path where chips will be written to disk; should be empty to begin with,
# after running once, should contain two folders -- 'train' and 'validation' --
# that each hold (a) a folder for each uuid in the analogous directory in 
# 'image_folder' and (b) a set of chips for the three channels -- vh, vv, and bathymetry --
# used by the reference implementation.
chips_path = '/home/jovyan/process_uuid'

# Set to true to overwrite preprocessing; set to false fo use existing preprocessed data
overwrite_preproc = False

# Paths defined in accordance with instructions above; should not need to change 
train_data_root = Path(image_folder) / 'train'
train_label_file = Path(label_file_root) / 'train.csv'
train_chips_path = Path(chips_path) / 'train'
val_data_root = Path(image_folder) / 'validation'
val_label_file = Path(label_file_root) / 'validation.csv'
val_chips_path = Path(chips_path) / 'validation'


# In[ ]:


# Create PyTorch datasets
channels = ['vh', 'vv', 'bathymetry']

train_data, val_data = create_datasets(
    train_data_root=train_data_root,
    train_detect_file=train_label_file,
    train_chips_path=train_chips_path,
    val_data_root=val_data_root,
    val_detect_file=val_label_file,
    val_chips_path=val_chips_path,
    overwrite_preproc=overwrite_preproc,
    channels=channels
)

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

coco= get_coco_api_from_dataset(train_data)
iou_types = ["bbox"]
coco_evaluator = CocoEvaluator(coco, iou_types)