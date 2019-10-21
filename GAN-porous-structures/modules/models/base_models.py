from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
import tensorflow as tf

class Generator(Model):
    def __init__(self, inputs, outputs = None):
        if outputs == None:
            model = self.__build(inputs)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

    def __build(self, z_dim):
  
        input_Z = Input(shape=(z_dim,))
        input_C = Input(shape=(1,))

        #combined = Concatenate()([input_Z, input_C])
        
        g = Dense(128 * 4 * 4)(input_Z)
        g = Reshape((4, 4, 128))(g)
  
        g = Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
        g = UpSampling2D()(g)

        g = Conv2DTranspose(128, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
        g = UpSampling2D()(g)
            
        g = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
        
        g = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
    
        g = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same')(g)
        img = Activation('tanh')(g)

        #return Model(inputs = [input_Z, input_C], outputs = img)
        return Model(inputs = input_Z, outputs = img)

class Discriminator(Model):
    def __init__(self, img_shape=None, inputs = None, outputs = None, alpha = 0.2, droprate = 0.2):
        if outputs == None:
            self.alpha = 0.2
            self.droprate = droprate
            model = self.__build(img_shape)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

        self.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    def __build(self, img_shape:tuple):

        input_img = Input(shape = img_shape)
        #input_C = Input(shape=(1,), name='Input_C')
    
        #d = Conv2D(32, kernel_size=1, strides = 1, padding='same', name='concat_layer')(input_img)
        #d = LeakyReLU(alpha = self.alpha)(d) 
        #d = AveragePooling2D()(d)
    
        d = Conv2D(32, kernel_size=1, strides = 1, padding='same', name='concat_layer')(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = Dropout(rate = self.droprate)(d)
        d = AveragePooling2D()(d)

        #d = Conv2D(64, kernel_size=3, strides = 1, padding='same')(d)
        #d = BatchNormalization()(d)
        #d = LeakyReLU(alpha = self.alpha)(d)
        #d = Dropout(rate = self.droprate)(d)

        d = Conv2D(64, kernel_size=3, strides = 1, padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = Dropout(rate = self.droprate)(d)
        d = AveragePooling2D()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        d = Dense(128)(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = Dropout(rate = self.droprate)(d)
    
        d = Dense(1, activation='sigmoid')(d)


        model = Model(inputs=input_img, outputs=d)
    
        return model

class GAN(Model):
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        model = self.__build(generator, discriminator)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss='binary_crossentropy', 
                     optimizer=Adam())

    def __build(self, generator, discriminator):

        #input_Z = Input(shape=(z_dim,)) 
        #input_C = Input(shape=(1,))

        img = generator(generator.inputs)#generator([input_Z, input_C])
    
        # Combined Generator -> Discriminator model
        classification = discriminator(img)
    
        model = Model(generator.inputs, classification)

        return model