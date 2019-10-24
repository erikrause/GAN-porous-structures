import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-amd', action='store_true', help='Enable PlaidML backend (for AMD devices)')
args = parser.parse_args()

#################
#ENABLE PLAIDML FOR AMD DEVICES:
if args.amd:
    import os
    import plaidml.keras
    plaidml.keras.install_backend()
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#################

#from modules.models import base_models
from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import numpy as np

# Start input image resolution
channels = 1
start_shape = (8,8,8,channels)
z_dim = 100
n_blocks = 5        # End_shape = start_shape*n_blocks

# Filters for each resolution block:
n_filters = {1: 64,
             2: 32,
             3: 16,
             4: 8}    

filter_sizes = {1: 3,
                2: 3,
                3: 5,
                4: 5}      

# 'directory_name/' (слэш только в конце названия папки!).
DIRECTORY = ''
#DATASET_DIR = 'datasets/berea/{}.png'  -   for png files iteration
DATASET_DIR = 'datasets/beadpack/beadpack.tif'
# Initialize dataset:

img_dims = len(start_shape) - 1
data_loader = DataLoader(DATASET_DIR, (500, 500, 500), is_tif=True, dims=img_dims)
  

# Build a models (если модели и логи лежат в папке History, то будут загружены с диска):
WEIGHTS_DIR = 'models-custom/'
model_handler = ModelHandler(DIRECTORY, start_shape, z_dim, n_blocks,  n_filters, filter_sizes, data_loader)#, WEIGHTS_DIR)


######################################
# MAIN LOOPv7
######################################

batch_size = 16
sample_interval = 100    # должно быть кратно итерациям
# Итерации на каждый слой:
n_fadein = np.array([0, 2500, 2000, 2000, 2000])
n_straight = np.array([4000-4000, 4000, 3000, 3000, 3000])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)
