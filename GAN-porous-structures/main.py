# import sys
# if 'plaidml' in sys.modules:
#
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-amd', action='store_true', help='Enable PlaidML backend (for AMD devices)')
#     args = parser.parse_args()
#
#     #################
#     #ENABLE PLAIDML FOR AMD DEVICES:
#     if args.amd:
#         import os
#         import plaidml
#         import plaidml.keras
#         plaidml.keras.install_backend()
#         os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#         #import keras
#     #################

#-----------------------------
# Jupyter notebook starts here:
#-----------------------------

# 'directory_name/' (слэш только в конце названия папки!).
DIRECTORY = ''
#DATASET_DIR = 'datasets/berea/{}.png'  -   for png files iteration
DATASET_DIR = 'datasets/berea/berea.tif'
is_tif = True      # Change to false for downloading .png files

# Uncomment this for notebook
#import sys
#sys.path.append(DIRECTORY)

from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import numpy as np

# For setting hyperparameters to model:
from modules.models import base_models
from keras import initializers
from keras.optimizers import Adam, RMSprop

# Start input image resolution
channels = 1        # 1 - черно-белое изображение
start_shape = (8,8,8,channels)
z_dim = 256         # Для сложных данных увеличить z_dim
n_blocks = 5        # Количество повышений разрешения. End_shape = start_shape*n_blocks
is_nearest = False   # Алгоритм понижения разрешения датасета. False = Avgpoolng: True = MaxPooling.

# Filters for each resolution block (from hidden layers to end resolution layers):
# !!! len(n_filters) == len(filter_sizes) !!!
n_filters = [ 64,   # 8x8x8     - example of resolution
              32,   # 16x16x16
              16,   # 32x32x32
              8,    # 64x64x64
              4 ]   # 128x128x128

filters_sizes = [ 3,     # 8x8x8     # TODO: попробовать большой фильтр
                  3,     # 16x16x16
                  3,     # 32x32x32
                  3,     # 64x64x64
                  3]     # 128x128x128

# Initialize dataset:
img_dims = len(start_shape) - 1

# Нужно чтобы dataset_shape делилось на 2^n_blocks-1 без остатка (обрезать датасет, если необходимо)
#dataset_shape = (496,466,496)   # for beadpack, original - (500, 500, 500)
dataset_shape = (400,400,400)   # for berea
#dataset_shape = (256,256,256)   # for ketton
data_loader = DataLoader(DATASET_DIR, dataset_shape, n_blocks, is_tif=is_tif, dims=img_dims, is_nearest_batch=is_nearest)

# Build a models (если логи лежат в папке History, то веса моделей будут загружены с папки History/models_wetights):
WEIGHTS_DIR = 'models-custom/'  # не используется

# Setting model layers parameters/
n_filters = np.asarray(n_filters)
n_filters = n_filters * 2 
base_models.n_filters = n_filters
base_models.filters_sizes = filters_sizes
base_models.conv_per_res = 1
# /

# Setting hyperparameters/
batch_size = 32
base_models.batch_size = None     # Set to None if you need dinamic batch_size. Else set to train batch_size.
base_models.lr = 0.0005
base_models.dis_lr = base_models.lr
base_models.opt = Adam(lr=base_models.lr)
base_models.dis_opt = Adam(lr=base_models.dis_lr)
# TODO: add dropout or noise for dis
base_models.alpha = 0.2     # ReLU alpha value
base_models.weight_init = initializers.he_normal() 
sample_interval = 100    # должно быть кратно итерациям
# /

model_handler = ModelHandler(DIRECTORY, start_shape, z_dim, n_blocks, n_filters, filters_sizes, data_loader)#, WEIGHTS_DIR)

######################################
# MAIN LOOP
######################################

# Итерации на каждый слой:
n_fadein = np.array([0, 4000, 4000, 4000, 4000])
n_straight = np.array([8000, 12000, 17000, 20000, 200000])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)
