import numpy as np
from numpy import random
from PIL import Image
import cv2

from keras.layers import Input
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.models import Model

class DataLoader(object):
  def __init__(self, directory:str, resolution:tuple):
    
    imageArray = np.empty((resolution[0],resolution[1],resolution[2]))
    #self.count = count
    self.resolution = resolution
    
    for i in range(0, resolution[0]): 
        filename = directory.format(i+1)
        image = Image.open(filename)
        nparray = np.array(image)
        imageArray[i] = nparray
        ############
        # 3D:
        #from skimage import io
        #im = io.imread('E:/Практика/beadpack/beadpack.tif')
        ############
        print('\r', i+1, ' image loaded.', end='')
    self.dataset = imageArray#.reshape(count,resolution,resolution,1)

    self.downsample_network = DownSamplingNetwork()
    
  def get_batch(self, batch_size:int, resolution:tuple, downscale:int):
    voxels = np.empty((batch_size, resolution[0],resolution[1],resolution[2]))
    
    for i in range(0, batch_size):
        z_value = random.randint(0, self.resolution[0]- 1 - resolution[0])
        x_value = random.randint(0, self.resolution[1] - 1 - resolution[1])
        y_value = random.randint(0, self.resolution[2] - 1 - resolution[2])
        voxels[i] = self.dataset[z_value:z_value + resolution[0],
                                     x_value:x_value + resolution[1], 
                                     y_value:y_value + resolution[2]]
    dz = resolution[0]//downscale
    dx = resolution[1]//downscale
    dy = resolution[2]//downscale

    voxels = voxels.reshape(batch_size, 128, 128, 128, 1)
    voxels = self.downsample_network.calculate(voxels, 8)

    voxels = voxels / 127.5 - 1.0
    return voxels

# Network, using for down sampling with linear? interpolation. (For nearest interpolation change AvgPool layer to MaxPool.
class DownSamplingNetwork():
    def __init__(self):
        voxel = Input(shape=(None,None,None,1))     # Variable input voxel shape.
        x = AveragePooling3D()(voxel)
        self.model = Model(inputs=voxel, outputs=x)

    def calculate(self, voxels, downscale:tuple):
        new_voxels = voxels
        i = 2

        while i <= downscale:
            new_voxels = self.model.predict(new_voxels)
            i = i*2

        return new_voxels