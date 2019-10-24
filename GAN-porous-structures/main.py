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

from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import numpy as np

# Start input image resolution
channels = 1        # 1 - черно-белое изображение
start_shape = (8,8,channels)
z_dim = 100         # Для сложных данных увеличить z_dim
n_blocks = 5        # Количество повышений разрешения. End_shape = start_shape*n_blocks

# Filters for each resolution block (from hidden layers to end resolution layers):
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
DATASET_DIR = 'datasets/beadpack/{}.png'
is_tif = False      # Change to false for downloading .png files

# Initialize dataset:
img_dims = len(start_shape) - 1
data_loader = DataLoader(DATASET_DIR, (500, 500, 500), is_tif=is_tif, dims=img_dims)

# Build a models (если логи лежат в папке History, то веса моделей будут загружены с папки History/models_wetights):
WEIGHTS_DIR = 'models-custom/'
model_handler = ModelHandler(DIRECTORY, start_shape, z_dim, n_blocks,  n_filters, filter_sizes, data_loader)#, WEIGHTS_DIR)

######################################
# MAIN LOOP
######################################

batch_size = 64
sample_interval = 100    # должно быть кратно итерациям
# Итерации на каждый слой:
n_fadein = np.array([0, 2500, 2000, 2000, 2000])
n_straight = np.array([4000, 4000, 3000, 3000, 3000])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)
