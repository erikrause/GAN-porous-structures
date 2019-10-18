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

# Start input image dimensions
channels = 1
start_shape = (8, 8, 8, channels)

#DIRECTORY = '/content/drive/My Drive/GAN/PGGANv5/beadpack'
#DATASET_DIR = '/content/drive/My Drive/GAN/datasets/beadpack'

# directory_name/ (слэш только в конце названия папки!).
DIRECTORY = ''
DATASET_DIR = DIRECTORY + 'datasets/beadpack/'
# Initialize dataset:
data_loader = DataLoader(DATASET_DIR+'/{}.png', (500,500,500))

# Size of the noise vector, used as input to the Generator
z_dim = 100
# Number of progressive resolution blocks:
n_blocks = 5    
# Filters for each resolution block:
n_filters = {1: 16,
             2: 16,
             3: 16,
             4: 8}    

filter_sizes = {1: 3,
                2: 3,
                3: 3,
                4: 5}
# Build a models (если модели и логи лежат в папке History, то будут загружены с диска):
#WEIGHTS_DIR = 'models-custom/'
model_handler = ModelHandler(DIRECTORY, start_shape, z_dim, n_blocks,  n_filters, filter_sizes, data_loader)#, WEIGHTS_DIR)

######################################
# MAIN LOOP
######################################

batch_size = 64
sample_interval = 100    # должно быть кратно итерациям
# Итерации на каждый слой:
#n_fadein = np.array([0, 3000, 8000, 10000])
#n_straight = np.array([1500, 8500, 2500, 2500])
n_fadein = np.array([0, 5000, 8000, 20000, 30000])
n_straight = np.array([40000, 4000, 10000, 10000, 25000])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)

#from keras.utils import plot_model
#for i in range(0, 4):
#    for j in range(0,2):
#        plot_model(models.generators[i][j], to_file='E:/prob/generators-{}-{}.pdf'.format(i,j))
#        plot_model(models.discriminators[i][j], to_file='E:/prob/discriminators-{}-{}.pdf'.format(i,j))

