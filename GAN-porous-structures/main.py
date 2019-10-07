#################
#FOR AMD DEVICES:
#import os
#import plaidml.keras
#plaidml.keras.install_backend()
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#################
#from modules.models import base_models
from modules.preprocess import DataLoader
from modules.ModelHandler import ModelHandler
import numpy as np

img_rows = 16
img_cols = 16
channels = 1
droprate = 0.2

# Start input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100

#DIRECTORY = '/content/drive/My Drive/GAN/PGGANv5/beadpack'
#DATASET_DIR = '/content/drive/My Drive/GAN/datasets/beadpack'

DIRECTORY = 'E:/Практика/beadpack'
DATASET_DIR = DIRECTORY + '/dataset'

data_loader = DataLoader(DATASET_DIR+'/{}.png', 500, (500,500))

n_blocks = 4

n_filters = {1: 64,
             2: 32,
             3: 16}

filter_sizes = {1: (3,3),
                2: (3,3),
                3: (3,3)}        ## Протестировать фильтры 5х5


model_handler = ModelHandler(DIRECTORY, img_shape, z_dim, n_blocks,  n_filters, filter_sizes, data_loader)

######################################
# MAIN LOOPv7
######################################

batch_size = 64
sample_interval = 100
# Итерации на каждый слой:
n_fadein = np.array([0, 6000, 8000, 10000])
n_straight = np.array([2500, 2500, 2500, 2500])

model_handler.train(n_straight, n_fadein, batch_size, sample_interval)

#from keras.utils import plot_model
#for i in range(0, 4):
#    for j in range(0,2):
#        plot_model(models.generators[i][j], to_file='E:/prob/generators-{}-{}.pdf'.format(i,j))
#        plot_model(models.discriminators[i][j], to_file='E:/prob/discriminators-{}-{}.pdf'.format(i,j))
