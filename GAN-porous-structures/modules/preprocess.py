import numpy as np
from numpy import random
from PIL import Image
from skimage import io  # for tiff load

from tensorflow.keras.layers import Input, MaxPooling3D, AveragePooling3D, MaxPooling2D, AveragePooling2D
#from keras.layers.convolutional import MaxPooling3D, AveragePooling3D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model

import math
from PIL import Image
import os

class DataLoader(object):
    def __init__(self, filename:str, resolution:tuple, n_blocks:int, is_tif = True, dims = 3, is_nearest_batch = False):
    

        self.resolution = resolution
        self.dims = dims

        if is_tif:
            self.dataset = self.__get_data_from_tif(filename)
        else:
            self.dataset = self.__get_data_from_pngs(resolution[0], filename)

        self.downsample_network = DownSamplingNetwork(dims, is_nearest_batch)
        self.dataset = self.dataset[:496,:496,:496]
        self.datasets = []

        self.dataset = np.expand_dims(self.dataset, axis=0)
        self.dataset = np.expand_dims(self.dataset, axis=-1)
        self.datasets.append(self.dataset)
        

        # debug:(gen_imgs+1)*127.5
        debug = (self.datasets[0][0,0,:,:,0]+1)*127.5
        debug = self.datasets[0][0,0,:,:,0]
        debug_sample = Image.fromarray(self.datasets[0][0,0,:,:,0])
        debug_path = 'debug/dataset/'
        os.makedirs(debug_path, exist_ok=True)
        debug_sample.save(debug_path + 'dataset-0.png')

        for i in range(1,n_blocks):
            self.datasets.append(self.downsample_network.calculate(self.datasets[-1], 2))
            # debug:
            debug = self.datasets[i][0,0,:,:,0].astype('uint8')
            debug_sample = Image.fromarray(debug)
            debug_sample.save('{}dataset-{}.png'.format(debug_path, i))
    
    def get_batch(self, batch_size:int, resolution:tuple, downscale:int):

        m = int(math.log(downscale, 2))
        images = []
        start_res = resolution[0] // downscale
        #tmp =[]
        for i in range(0, batch_size):
            value = []
            for axis in range(0, len(resolution)):
                #value.append(random.randint(0, self.resolution[axis]- 1 - resolution[axis]))
                value.append(random.randint(0, self.datasets[m].shape[axis+1]-1-(resolution[axis]//downscale)))
                #z_value = random.randint(0, self.resolution[0]- 1 - resolution[0])
                #x_value = random.randint(0, self.resolution[1] - 1 - resolution[1])
                #y_value = random.randint(0, self.resolution[2] - 1 - resolution[2])

            if self.dims == 3:
                images.append(self.datasets[m][0, 
                                               value[0]:value[0] + resolution[axis]//downscale,
                                               value[1]:value[1] + resolution[axis]//downscale, 
                                               value[2]:value[2] + resolution[axis]//downscale,
                                               0])
            elif self.dims == 2:
                image_number = random.randint(0, self.resolution[0])
                images.append(self.datasets[m][0,
                                               image_number,
                                               value[0]:value[0] + resolution[axis]//downscale, 
                                               value[1]:value[1] + resolution[axis]//downscale,
                                               0])
            #if len(tmp) > 128:
            #  tmp = np.asarray(tmp)
            #  tmp = np.expand_dims(tmp, axis=-1)
            #  tmp = self.downsample_network.calculate(tmp, downscale)
            #  images.extend(tmp)
            #  tmp = []
        #tmp = np.asarray(tmp)
        #tmp = np.expand_dims(tmp, axis=-1)
        #tmp = self.downsample_network.calculate(tmp, downscale)
        #images.extend(tmp)
        #self.debug(images[i,:,:])
        images = np.asarray(images)
        images = np.expand_dims(images, axis=-1)

        #images = self.downsample_network.calculate(images, downscale)
        images = images / 127.5 - 1.0

        return images
    

    def __get_data_from_pngs(self, count, filename):
        imageArray = np.empty((self.resolution[0],self.resolution[1],self.resolution[2]))
        for i in range(0, count):
            filename = filename.format(i+1)
            image = Image.open(filename)
            nparray = np.array(image)
            imageArray[i] = nparray
            print('\r', i+1, ' image loaded.', end='')
        return imageArray

    def __get_data_from_tif(count, filename):
        return io.imread(filename)



# Network, using for down sampling with linear? interpolation. (For nearest interpolation change AvgPool layer to MaxPool.
class DownSamplingNetwork():
    def __init__(self, dims, is_nearest=False):
        start_shape = [1]
        for i in range(0, dims):
            start_shape.insert(0, None)
        shape = list(start_shape)
        
        voxel = Input(shape=shape)

        if dims == 3:
            if is_nearest:
                x = MaxPooling3D()(voxel)
            else:
                x = AveragePooling3D()(voxel)
        elif dims == 2:
            if is_nearest:
                x = MaxPooling2D()(voxel)
            else:
                x = AveragePooling2D()(voxel)

        self.model = Model(inputs=voxel, outputs=x)

    def calculate(self, imgs, downscale):

        new_imgs = imgs
        i = 2
        while i <= downscale:
            new_imgs = self.model.predict(new_imgs)
            i = i*2

        return new_imgs