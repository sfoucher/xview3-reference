#!/usr/bin/env python
# coding: utf-8

# # Training the reference xView3 model
# 
# This tutorial is heavily based on the [Pytorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision\_tutorial.html).  As such, it is not intended to represent a *good* approach to the xView3 problem, but rather how a standard approach might be modified to use the xView3 dataset.

# In[ ]:


#get_ipython().system('pip install ray rasterio pycocotools')


# ## Imports

# In[8]:


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
overwrite_preproc = True

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
if 0:
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
else:
    train_data= torch.load(tiny_data_path / 'train_data.pth')
    val_data= torch.load(tiny_data_path / 'val_data.pth')

# ## Data loaders
# Now that our data is loaded and initialized, we want to start setting up the device and DataLoaders to feed the PyTorch model.
# 
# Depending on your GPU vRAM, play around with the batch_size for the training and validation data loaders to ensure you don't run into out of memory errors.

# In[ ]:


np.seterr(divide='ignore', invalid='ignore')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# define training and validation data loaders; can increase number of workers
# for speed 
train_sampler = torch.utils.data.RandomSampler(train_data)
val_sampler = torch.utils.data.SequentialSampler(val_data)

data_loader_train = torch.utils.data.DataLoader(
    train_data, batch_size=8, sampler=train_sampler, num_workers=0,
    collate_fn=collate_fn, pin_memory=True)

data_loader_val = torch.utils.data.DataLoader(
    val_data, batch_size=8, sampler=val_sampler, num_workers=0,
    collate_fn=collate_fn, pin_memory=True)


# ## Visualizing the data inside the data loader

# In[ ]:


# Get a batch of data from the validation data loader
batch = iter(data_loader_val).next()
print(batch[0][0].shape)
print(batch[1][0])


# In[7]:
# %%

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a 10x10 figure
plt.figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(10,10))

# Display the image
im = batch[0][0][0,:,:]
pcm = ax.imshow(im, cmap='bone')
plt.colorbar(pcm, ax=ax)

# Plot the detections on top of the image
for box in batch[1][0]['boxes']:
    rect = patches.Rectangle((box[0], box[1]), 10, 10, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
plt.savefig('file_name.png')
plt.show()

# %%
# ## Model
# The xView3 reference model utilizes the well-known [Faster R-CNN](https://arxiv.org/abs/1506.01497) model that is provided pre-built in the torchvision library. We wrap the Faster R-CNN into a convenience class for ease of use, but the basic structure still remains and should be simple to extend.
# 
# <img src="figures/faster_rcnn.jpeg" alt="Faster R-CNN model structure" style="height:450px;">
# 
# The xView3 Faster R-CNN model is instaniated with 3 classes -- `not vessel`, `fishing`, and `not fishing` -- and a pre-defined image mean and standard deviation.  We also -- very naively! -- assume that every detection has a bounding box of the same size.  While validation data for this challenge has bounding boxes, the training data does not.  How you make use of these pieces of information is one of the ways in which you can be creative!

# In[8]:


# Getting and saving estimated image means/std needed for Faster RCNN implementation; 
if (not os.path.exists(f'{chips_path}/data_means.npy')) or (overwrite_preproc):
    image_mean = [0.5] * (len(train_data.channels))
    image_std = [0.1] * (len(train_data.channels))
    np.save(f'{chips_path}/data_means.npy', image_mean)
    np.save(f'{chips_path}/data_std.npy',image_std)
else:
    image_mean = np.load(f'{chips_path}/data_means.npy')
    image_std = np.load(f'{chips_path}/data_std.npy')

# Instantiating a Faster-RCNN model for maritime object detection/classification
model = xView3BaselineModel(num_classes=len(train_data.label_map.keys()), 
                            num_channels=len(train_data.channels),
                            image_mean=image_mean,
                            image_std=image_std)

# Move model to the right device
model.to(device)


# ### Optimizer and training parameters
# We use a standard stochastic gradient descent optimizer for the Faster R-CNN model. Feel free to play around with more advanced optimizers such as Adam, RMSProp, or others.
# 
# Additionally, we use a step-based learning rate scheduler to decay the learning rate over the duration of the training. You could experiment with other LR schedules.  For the reference implementation, we've kept this straightforward.

# In[9]:


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0075,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# let's train it for N epochs
num_epochs = 3


# ### Train the model
# Training the model is done via torchvision's provided `train_one_epoch` function which wraps the traditional PyTorch training loop into an easy-to-use function made for built-in object detectors. The model is saved every epoch.

# In[10]:


# Train the model for three epochs
for epoch in range(num_epochs):
    # train for one epoch, printing every iteration
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_val, device=device)

    checkpoint_path = f'trained_model_{epoch+1}_epochs.pth'
    torch.save(model.state_dict(), checkpoint_path)


# # Inference
# Let's run inference on the model by re-using one of the validation scenes

# In[11]:


# Creating a fresh model object for evaluation
model_eval = xView3BaselineModel(
            num_classes=len(train_data.label_map.keys()), 
            num_channels=len(train_data.channels),
            image_mean=image_mean,
            image_std=image_std,
)

# Creating a fresh dataloader for evaluation that includes *all* chips, not just labeled ones
# Note -- we rechip in-m
from dataloader import XView3Dataset
from utils import collate_fn
val_data_unlabeled = XView3Dataset(
        val_data_root,
        None,
        "test",
        detect_file=None,
        scene_list=val_data.scenes,
        chips_path=chips_path,
        background_frac=0.0,
        overwrite_preproc=False,
        channels=channels,
    )
    
data_loader_val_unlabeled = torch.utils.data.DataLoader(
        val_data_unlabeled,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

# Loading our last checkpoint
model_eval.load_state_dict(torch.load(f'trained_model_{num_epochs}_epochs.pth', map_location=device))

# Putting model on correct device
model_eval.to(device)

# Putting model into evaluation mode
model_eval.eval()


# Note that to run inference for the challenge, we must not only run the model forward -- we must also make sure that the output is formatted in the manner expected by the xView3 metric computation code.  The specification for this output can be found in the challenge documentation.

# In[12]:


from tqdm import tqdm
from constants import PIX_TO_M, FISHING, NONFISHING

res = {}

# Running inference
with torch.no_grad():
    for image, targets in tqdm(
        data_loader_val, total=len(data_loader_val_unlabeled)
    ):
        image = list(img.to(device) for img in image)
        targets = [
            {
                k: (v.to(device) if not isinstance(v, str) else v)
                for k, v in t.items()
            }
            for t in targets
        ]
        # Only strictly required for multi-GPU implementations 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Running model forward and moving outputs to cpu
        outputs = model_eval(image)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        # Translating from bounding boxes to point predictions
        for jj, (target, output) in enumerate(zip(targets, outputs)):
            output.update(
                {
                    "pred_chip_columns": [
                        [int(np.mean([box[0], box[2]])) for box in output["boxes"]]
                        if len(output["boxes"]) > 0
                        else []
                    ]
                }
            )
            output.update(
                {
                    "pred_chip_rows": [
                        [int(np.mean([box[1], box[3]])) for box in output["boxes"]]
                        if len(output["boxes"]) > 0
                        else []
                    ]
                }
            )

            # Estimating length using the size of the bounding box; note that because
            # uniformly sized bouding boxes were used in the reference implementation,
            # we don't expect this to work very well!
            
            pix_to_m = PIX_TO_M
            lengths = [
                ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
                for xmin, ymin, xmax, ymax in output["boxes"]
            ]
            lengths = [float(pix_to_m) * l for l in lengths]
            ratio = [
                (xmax - xmin) / (ymax - ymin)
                for xmin, ymin, xmax, ymax in output["boxes"]
            ]

            output.update({"lengths": lengths})

        batch_res = {
            int(target["image_id"].cpu()): {
                "boxes": output["boxes"],
                "lengths": output["lengths"],
                "labels": output["labels"],
                "scores": output["scores"],
                "scene_id": target["scene_id"],
                "chip_id": int(target["chip_id"].cpu()),
                "pred_chip_columns": output["pred_chip_columns"][0],
                "pred_chip_rows": output["pred_chip_rows"][0],
            }
            for target, output in zip(targets, outputs)
        }

        res.update(batch_res)

# Instantiate output dataframe in xView3-specified format
df_out = pd.DataFrame(
    columns=(
        "detect_scene_row",
        "detect_scene_column",
        "scene_id",
        "is_vessel",
        "is_fishing",
        "vessel_length_m",
    )
)

# Formatting output in xView3 format; converting chip-level detection
# coordinates to scene-level detection coordinates
for inf_img_id, inf_val in tqdm(res.items()):
    for idx, box in enumerate(inf_val["boxes"]):
        # x, y = center(box)
        label = inf_val["labels"][idx].item()
        is_fishing = label == FISHING
        is_vessel = label in [FISHING, NONFISHING]
        length = inf_val["lengths"][idx].item()
        score = inf_val["scores"][idx].item()
        scene_id = inf_val["scene_id"]
        chip_id = inf_val["chip_id"]

        # Getting chip transforms
        with open(f"{chips_path}/validation/{scene_id}/coords.json") as fl:
            coords = json.load(fl)
        chip_offset_col, chip_offset_row = coords["offsets"][chip_id]

        # Adjusting chip pixel preds to global scene pixel preds
        chip_pred_row = inf_val["pred_chip_rows"][idx]
        chip_pred_column = inf_val["pred_chip_columns"][idx]
        scene_pred_row = chip_pred_row + chip_offset_row
        scene_pred_column = chip_pred_column + chip_offset_col

        df_out.loc[len(df_out)] = [
            scene_pred_row,
            scene_pred_column,
            scene_id,
            is_vessel,
            is_fishing,
            length,
        ]

print(f"{len(df_out)} detections found")
df_out.to_csv('/tmp/inference_output.csv', index=False)
print(df_out)


# # Metrics

# In order to properly detect and classify dark vessel activity in oceans, the xView3 challenge incorporates multiple different factors when scoring how well a model performs for any scene provided for inference. This aggregate metric takes into account five different components, explained below.
# 
# $$ M_R = F1_D * \frac{1 + F1_S  + F1_V + F1_F + PE_L}{5} $$
# 
# $$ PE_{L} = 1 - \frac{1}{N} \sum_{n=1}^{N} \frac{| \hat{\ell} - \ell|}{\ell}. $$
# 
# ## Maritime Object Detection ($F1_D$)
# The maritime object detection score, $F1_D$, is the Dice coefficient on the core object detection challenge presented by xView3 -- identifying vessels and fixed infrastructure in each scene. Ground truth positives are considered to be man-made maritime objects that are validated by a human labeler and/or [Automatic Identification System (AIS)](https://en.wikipedia.org/wiki/Automatic_identification_system) correlation.
# 
# In order to perform highly on this component, a model must by able to map the shoreline to a high fidelity, handle SAR artifacts such as ambiguities and sea clutter, and generalize well across a variety of sea states, [satellite sensor configurations](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/stripmap), and geographic regions.
# 
# ## Close-to-Shore Vessel Detection ($F1_S$)
# Detecting vessels close to the shore is of vital interest to the IUU enforcement community. Since vessels tend to be grouped close together near the shore, and land-based artifacts such as boulders and radar reflections are common, the detection task is harder. "Close to shore" vessels are defined as those vessels up to 2km away from the shoreline.
# 
# ## Vessel Classification ($F1_V$)
# Classifying a maritime object as "vessel" or "non-vessel" is the lowest fidelity classification that is useful to the IUU enforcement community. Positive classes are defined as "vessel" upon which an F1 score is calculated with regards to the ground truth.
# 
# ## Fishing Classification ($F1_F$)
# Once you have identified that a detected object is a "vessel", an F1 score is calculated over the model's ability to differentiate between "fishing" and "non-fishing" vessels, with the positive class being "fishing". There are many different types of fishing vessels, such as trawlers, trollers, dredge fishers, and longliners. A complete list is available in `dataloader.py`'s `get_label_map()` function.
# 
# ## Vessel Length Estimation ($PE_L$)
# Knowing the size of the vessel is critical for IUU enforcement efforts. Since SAR is inherently noisy, this component is scored using an aggregate percent error defined as $PE_L$ above.
# 
# These functions are implemented in `metric.py`.  Note that you may need to update the `shore_root` path below to reflect the path to your reference implementation data.

# In[15]:


from metric import score

inference = pd.read_csv('/tmp/inference_output.csv', index_col=False)
ground_truth = pd.read_csv(val_label_file, index_col=False)

ground_truth = ground_truth[
    ground_truth["scene_id"].isin(inference["scene_id"].unique())
].reset_index()

score_all = False
# By default we only score on high and medium confidence labels
if not score_all:
    ground_truth = ground_truth[
        ground_truth['confidence'].isin(['HIGH', 'MEDIUM'])
    ].reset_index()

out = score(inference, ground_truth, shore_root='/home/ubuntu/data/shoreline',
           distance_tolerance=200, shore_tolerance=2)
print(out)


# # Visualize predictions

# In[16]:


import matplotlib as mpl
from matplotlib import pyplot as plt
import rasterio

vmin, vmax = -35, -5

def display_image_in_actual_size(im_data, rows, cols, rows2=None, cols2=None):

    dpi = mpl.rcParams['figure.dpi']
    height, width= im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    print('Plotting image...')
    ax.matshow(im_data, cmap="bone", vmin=vmin, vmax=vmax)

    print('Plotting scatterplot...')
    ax.scatter(cols, rows, s=120, facecolors="none", edgecolors="r")
    if rows2 is not None:
        ax.scatter(cols2, rows2, s=80, facecolors="none", edgecolors="g")

    plt.margins(0,0)
    plt.show()


# In[17]:


# Scene ID
scene_id = '590dd08f71056cacv'

# Getting detect df
df = pd.DataFrame()
detects = pd.read_csv(val_label_file)
df = pd.concat((df, detects))
df.rename({'detect_scene_row':'scene_rows', 'detect_scene_column':'scene_cols'}, inplace=True, axis='columns')

df = df[df['scene_id'] == scene_id]

# Loading image in UTM coordinates
grdfile = val_data_root / scene_id / 'VH_dB.tif'
src = rasterio.open(grdfile)
image_orig = src.read(1)

# Getting predictions from file
preds_path = '/tmp/inference_output.csv'
df_preds = pd.read_csv(preds_path)
# Renaming columns for convenience
df_preds.rename({'detect_scene_row':'scene_rows', 'detect_scene_column':'scene_cols'}, inplace=True, axis='columns')

# Identifying a detection on which to center plot
row = df.iloc[-3]
print(row)

# Defining size of image patch for plot (generally suggest keeping ~ 1000)
patch_half_width = 750

# Getting predictions and detection sin image patch 
df_small = df[np.abs(df.scene_rows - row.scene_rows) < patch_half_width]
df_small = df_small[np.abs(df_small.scene_cols - row.scene_cols) < patch_half_width]

df_preds_small = df_preds[np.abs(df_preds.scene_rows - row.scene_rows) < patch_half_width]
df_preds_small = df_preds_small[np.abs(df_preds_small.scene_cols - row.scene_cols) < patch_half_width]

dt = image_orig[row.scene_rows-(patch_half_width):row.scene_rows+(patch_half_width), row.scene_cols-(patch_half_width):row.scene_cols+(patch_half_width)]

# Plotting ground truth (red) and predicted (green) detections
display_image_in_actual_size(dt, 
                            df_small.scene_rows - row.scene_rows + patch_half_width,
                            df_small.scene_cols - row.scene_cols + patch_half_width,
                            rows2=df_preds_small.scene_rows - row.scene_rows + patch_half_width,
                            cols2=df_preds_small.scene_cols - row.scene_cols + patch_half_width,
                            )


# In[ ]:




