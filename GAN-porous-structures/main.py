import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-amd', action='store_true', help='Enable PlaidML backend (for AMD devices)')
args = parser.parse_args()

#################
#ENABLE PLAIDML FOR AMD DEVICES:
if args.amd:
    import os
    import plaidml
    import plaidml.keras
    plaidml.keras.install_backend()
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    #import keras
#################

from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import numpy as np

# For setting hyperparameters to model:
from modules.models import base_models
from keras.optimizers import Adam, RMSprop
from keras import initializers

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

# 'directory_name/' (слэш только в конце названия папки!).
DIRECTORY = ''
#DATASET_DIR = 'datasets/berea/{}.png'  -   for png files iteration
DATASET_DIR = 'datasets/beadpack/beadpack.tif'
is_tif = True      # Change to false for downloading .png files

# Initialize dataset:
img_dims = len(start_shape) - 1
data_loader = DataLoader(DATASET_DIR, (496,466,496), n_blocks, is_tif=is_tif, dims=img_dims, is_nearest_batch=is_nearest)

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
base_models.batch_size = 32     # set to None if you need dinamic batch_size
base_models.lr = 0.0005
base_models.dis_lr = base_models.lr
base_models.opt = Adam(lr=base_models.lr)
base_models.dis_opt = Adam(lr=base_models.dis_lr)
# TODO: add dropout or noise for dis
base_models.alpha = 0.2     # ReLU alpha value
base_models.weight_init = initializers.he_normal() 
sample_interval = 1    # должно быть кратно итерациям
# /


######################################
# MAIN LOOP
######################################

model_handler = ModelHandler(DIRECTORY, start_shape, z_dim, n_blocks, n_filters, filters_sizes, data_loader)#, WEIGHTS_DIR)
# Итерации на каждый слой:
#n_fadein = np.array([0, 2500, 3500, 10000, 14000])
#n_straight = np.array([3400, 6000, 30000, 80000, 200000])
n_fadein = np.array([0, 5000, 5000, 50000, 14000])
n_straight = np.array([0, 5900, 50000, 80000, 200000])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)
