import numpy as np
from numpy import random
from PIL import Image
from skimage import io  # for tiff load

from keras.layers import Input
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D, MaxPooling2D, AveragePooling2D
from keras.models import Model

class DataLoader(object):
    def __init__(self, filename:str, resolution:tuple, is_tif = True, dims = 3, is_nearest_batch = False):
    
        imageArray = np.empty((resolution[0],resolution[1],resolution[2]))
        #self.count = count
        self.resolution = resolution
        self.dims = dims

        if is_tif:
            self.dataset = self.__get_data_from_tif(filename)
        else:
            self.dataset = self.__get_data_from_pngs(resolution[0], filename)
        

        self.downsample_network = DownSamplingNetwork(dims, is_nearest_batch)
    
    def get_batch(self, batch_size:int, resolution:tuple, downscale:int):
  
        images = []
        for i in range(0, batch_size):
            value = []
            for axis in range(0, len(resolution)):
                value.append(random.randint(0, self.resolution[axis]- 1 - resolution[axis]))
                #z_value = random.randint(0, self.resolution[0]- 1 - resolution[0])
                #x_value = random.randint(0, self.resolution[1] - 1 - resolution[1])
                #y_value = random.randint(0, self.resolution[2] - 1 - resolution[2])

            if self.dims == 3:
                #images = np.empty((batch_size, resolution[0],resolution[1],resolution[2]))
                images.append(self.dataset[value[0]:value[0] + resolution[0],
                                             value[1]:value[1] + resolution[1], 
                                             value[2]:value[2] + resolution[2]])
            elif self.dims == 2:
                image_number = random.randint(0, self.resolution[0])
                images.append(self.dataset[image_number,
                                         value[0]:value[0] + resolution[0], 
                                         value[1]:value[1] + resolution[1]])
                #self.debug(images[i,:,:])
        images = np.asarray(images)
        images = np.expand_dims(images, axis=-1)
        images = self.downsample_network.calculate(images, downscale)
        images = images / 127.5 - 1.0

        return images

    def __get_data_from_pngs(count, filename):
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