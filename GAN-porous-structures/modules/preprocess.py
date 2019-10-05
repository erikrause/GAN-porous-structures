import numpy as np
from numpy import random
from PIL import Image
import cv2

class DataLoader(object):
  def __init__(self, directory:str, count:int, resolution:tuple):
    
    imageArray = np.empty((count,resolution[0],resolution[1]))
    self.count = count
    self.resolution = resolution
    
    for i in range(0, count): 
        filename = directory.format(i+1)
        image = Image.open(filename)
        nparray = np.array(image)
        imageArray[i] = nparray   
        print('\r', i+1, ' image loaded.', end='')
    self.dataset = imageArray#.reshape(count,resolution,resolution,1)
    
  def get_batch(self, batch_size, resolution, downscale):
    imageArray = np.empty((batch_size, resolution[0], resolution[1]))
    
    for i in range(0, batch_size):
        image_number = random.randint(0, self.count)
        x_value = random.randint(0, self.resolution[0] - 1 - resolution[0])
        y_value = random.randint(0, self.resolution[1] - 1 - resolution[1])
        imageArray[i] = self.dataset[image_number, 
                                x_value:x_value + resolution[0], 
                                y_value:y_value + resolution[1]]
        
    dx = resolution[0]//downscale
    dy = resolution[1]//downscale
    if downscale > 1:
        imageArray = imageArray.transpose((1,2,0))
        imageArray = cv2.resize(imageArray, 
                                    dsize=(dx,dy), 
                                    interpolation=cv2.INTER_LINEAR)
        imageArray = imageArray.transpose((2,0,1))
    imageArray = imageArray.reshape(batch_size, dx, dy, 1)
    imageArray = imageArray / 127.5 - 1.0
    return imageArray