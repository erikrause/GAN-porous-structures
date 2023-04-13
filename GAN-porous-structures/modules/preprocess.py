import numpy as np
from numpy import random
from PIL import Image
from skimage import io  # for tiff load

from keras.layers import Input
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D, MaxPooling2D, AveragePooling2D
from keras.models import Model

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
        # self.dataset = self.dataset[:496,:496,:496]
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
        if debug_sample.mode != 'RGB':
            debug_sample = debug_sample.convert('RGB')
        debug_sample.save(debug_path + 'dataset-0.png')

        # Подготовить датасет для разных разрешений, если 3D. 2D downsampling высчитываются в get_batch.
        if dims == 3:
            for i in range(1,n_blocks):
                self.datasets.append(self.downsample_network.calculate(self.datasets[-1], 2))
                # debug:
                debug = self.datasets[i][0,0,:,:,0].astype('uint8')
                debug_sample = Image.fromarray(debug)
                debug_sample.save('{}dataset-{}.png'.format(debug_path, i))
    
    def get_batch(self, batch_size:int, resolution:tuple, downscale:int):

        if self.dims == 3:
            m = int(math.log(downscale, 2))
        else:
            m = 0

        images = []
        start_res = resolution[0] // downscale
        #tmp =[]
        for i in range(0, batch_size):
            value = []
            for axis in range(0, len(resolution)):
                #value.append(random.randint(0, self.resolution[axis]- 1 - resolution[axis]))
                if self.dims == 3:
                    value.append(random.randint(0, self.datasets[m].shape[axis+1]-1-(resolution[axis]//downscale)))
                elif self.dims == 2:
                    value.append(random.randint(0, self.datasets[m].shape[axis+1]-1-(resolution[axis])))
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
                images.append(self.datasets[0][0,
                                               image_number,
                                               value[0]:value[0] + resolution[axis],#//downscale, 
                                               value[1]:value[1] + resolution[axis],#//downscale,
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

        if self.dims == 2:
            images = self.downsample_network.calculate(images, downscale)
        images = (images / 127.5) - 1

        images = self.rotate(images)
        images = self.mirror(images)

        return images
    
    def rotate(self, images):

      if self.dims == 3:
        n_axes = np.random.randint(0, 4)
      elif self.dims == 2:
        n_axes = np.random.randint(0, 2)

      if n_axes == 0:
        return images   # return original

      else:
        rot_times = np.random.randint(0, 3)   # 4 times of rotation

        for i in range(0, rot_times):
          if n_axes == 1:
            axes = (1,2)
          elif n_axes == 2:
            axes = (2,3)
          elif n_axes == 3:
            axes = (1,3)
          
          images = np.rot90(images,axes=axes)

          # get axes for next rotations:
          if self.dims == 3:
            n_axes = np.random.randint(1, 4)
          elif self.dims == 2:
            n_axes = 1 # np.random.randint(1, 2)

        return images

    def mirror(self, images):
      
      if self.dims == 3:
        mir_times = np.random.randint(0, 3)
        axis = np.random.choice(4, mir_times + 1)
      elif self.dims == 2:
        mir_times = np.random.randint(0, 2)
        axis = np.random.choice(3, mir_times + 1)

      if axis[0] == 0:
        return images   # return original

      else:
        for i in range(0, len(axis)):
          images = np.flip(images, axis=axis[i])

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