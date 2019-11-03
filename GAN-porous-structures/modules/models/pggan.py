from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
import tensorflow as tf

import tensorflow.python.keras.backend as K

from modules.models import base_models

class WeightedSum(Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = backend.variable(alpha, name='ws_alpha')

	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2)
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

def update_fadein(models, step, n_steps, alpha = -1):
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                # calculate current alpha (linear from 0 to 1)
                if alpha == -1:
                    current_alpha = backend.eval(layer.alpha)#K.get_value(layer.alpha)
                    remaining_alpha = 1 - current_alpha
                    remaining_steps = n_steps - step
                    if remaining_steps == 0:
                        alpha = 1.0
                    else:
                        #alpha = step / float(n_steps - 1)
                        dalpha = remaining_alpha / remaining_steps
                        alpha = current_alpha + dalpha
                #alpha = 1       #debug
                backend.set_value(layer.alpha, alpha)
    return alpha

def add_block(old_model, n_filters=64, filter_size=3):
    models = []
    if isinstance(old_model, base_models.Discriminator):
        models = __add_discriminator_block(old_model, n_filters, filter_size)
    elif isinstance(old_model, base_models.Generator):
        models = __add_generator_block(old_model, n_filters, filter_size)
    elif isinstance(old_model, list):
        discriminators = old_model[0]
        generators = old_model[1]
        models = __add_gan_block(discriminators, generators)

    return models

def __add_generator_block(old_model, n_filters=64, filter_size=3):
    # get the end of the last block
    block_end = old_model.layers[-3].output
    conv = old_model.conv
    upsample = old_model.upsample

    n_filters = n_filters
    # upsample, and define new block
    #upsampling = upsample()(block_end)
    g = conv(n_filters, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=base_models.weight_init)(block_end)
    g = BatchNormalization()(g)
    g = ReLU()(g)
    g = upsample()(g)

    #g = conv(n_filters, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=base_models.weight_init)(g)
    #g = BatchNormalization()(g)
    #g = ReLU()(g)
    #g = upsample()(g)
    
    # add new output layer
    g = conv(1, kernel_size=3, strides=1, padding='same', kernel_initializer=base_models.weight_init)(g)
    out_image = Activation('tanh')(g)
    # define model
    straight_model = base_models.Generator(inputs=old_model.inputs,
                                           start_img_shape=old_model.start_img_shape,
                                           outputs=out_image)
    
    # get the output layer from old model
    #out_old = old_model.layers[-2]#[-1]
    # connect the upsampling to the old output layer
    #out_old = out_old(upsampling)
    out_image2 = upsample()(old_model.layers[-1].output)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    fadein_model = base_models.Generator(inputs=old_model.inputs,
                                         start_img_shape=old_model.start_img_shape,
                                         outputs=merged)
    return [straight_model, fadein_model]

def __add_gan_block(discriminators, generators):
    fadein_model = base_models.GAN(generators[1], discriminators[1])
    straight_model = base_models.GAN(generators[0], discriminators[0])
    
    return [straight_model, fadein_model]

def __add_critic_block(old_model, n_input_layers=5, n_filters=64, filter_size=3):
    #old_input_shape = list(old_model.input_shape) #get_input_shape_at(0)
    old_input_shape = list(old_model.get_input_shape_at(0))
    old_img_shape = old_input_shape[1:-1]
    new_img_shape = tuple(2*x for x in old_img_shape)   #shape without channels
    new_img_shape = list(new_img_shape)       
    new_img_shape.append(1)                             #add chanel shape
    new_img_shape = tuple(new_img_shape)   

    #input_img_shape = (old_input_shape[0][-2]*2, 
                   #old_input_shape[0][-2]*2, 
                   #old_input_shape[0][-1])      # Need to refactor
    input_img = Input(shape=new_img_shape)

    conv = old_model.conv
    pool = old_model.pool
    
    # New block/
    print(n_filters)
    
    d = conv(n_filters, 
               kernel_size=filter_size, 
               strides=1, 
               padding='same', 
               kernel_initializer=base_models.weight_init, 
               kernel_constraint=base_models.constraint)(input_img)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.02)(d)
    #d = Dropout(rate = 0.2)(d)
    d = pool()(d)   

    n_filters_last = old_model.layers[1].filters  #количество старых фильтров входа
    kernel_size_last = old_model.layers[1].kernel_size
    d = conv(n_filters_last,
               kernel_size = kernel_size_last, 
               strides=1, padding='same', 
               kernel_initializer=base_models.weight_init,
               kernel_constraint=base_models.constraint)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.02)(d)
    #d = Dropout(rate = 0.2)(d)
    d = pool()(d)   
    
    block_new = d
    #/New block
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)

        prob = current_layer.get_weights()
        
    straight_model = base_models.Critic(img_shape=new_img_shape,
                                        inputs=input_img, 
                                        outputs=d)

    downsample = pool()(input_img)
    
    block_old = downsample
    for i in range(1, n_input_layers):
        block_old = old_model.layers[i](block_old)
    
    d = WeightedSum()([block_old, block_new])
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)
        
    fadein_model = base_models.Critic(img_shape=new_img_shape, 
                                      inputs=input_img, 
                                      outputs=d)

    return [straight_model, fadein_model]

def __add_discriminator_block(old_model, n_filters=64, filter_size=3, n_input_layers=6,):
    #old_input_shape = list(old_model.input_shape) #get_input_shape_at(0)
    old_input_shape = list(old_model.get_input_shape_at(0))
    old_img_shape = old_input_shape[1:-1]
    new_img_shape = tuple(2*x for x in old_img_shape)   #shape without channels
    new_img_shape = list(new_img_shape)       
    new_img_shape.append(1)                             #add chanel shape
    new_img_shape = tuple(new_img_shape)   

    #input_img_shape = (old_input_shape[0][-2]*2, 
                   #old_input_shape[0][-2]*2, 
                   #old_input_shape[0][-1])      # Need to refactor
    input_img = Input(shape=new_img_shape)

    conv = old_model.conv
    pool = old_model.pool
    
    # New block/
    print(n_filters)
    
    d = conv(n_filters, 
               kernel_size=filter_size, 
               strides=1, 
               padding='same', 
               kernel_initializer=base_models.weight_init)(input_img)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.02)(d)
    d = Dropout(rate = 0.2)(d)
    d = pool()(d)

    n_filters_last = old_model.layers[1].filters  #количество старых фильтров входа
    kernel_size_last = old_model.layers[1].kernel_size

    #d = conv(n_filters_last, 
    #           kernel_size=filter_size, 
    #           strides=1, 
    #           padding='same', 
    #           kernel_initializer=base_models.weight_init)(d)
    #d = BatchNormalization()(d)
    #d = LeakyReLU(alpha=0.02)(d)
    #d = Dropout(rate = 0.2)(d)
    #d = pool()(d)   

    
    d = conv(n_filters_last,
               kernel_size = kernel_size_last, 
               strides=1, padding='same', 
               kernel_initializer=base_models.weight_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.02)(d)
    d = Dropout(rate = 0.2)(d)
    d = pool()(d)   
    
    block_new = d
    #/New block
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)

        prob = current_layer.get_weights()
        
    straight_model = base_models.Discriminator(img_shape=new_img_shape,
                                        inputs=input_img, 
                                        outputs=d)

    downsample = pool()(input_img)
    
    block_old = downsample
    for i in range(1, n_input_layers):
        block_old = old_model.layers[i](block_old)
    
    d = WeightedSum()([block_old, block_new])
    
    for i in range(n_input_layers, len(old_model.layers)):
        current_layer = old_model.layers[i]
        print(current_layer)
        if current_layer.name == 'Input_C':
            input_C = current_layer.input
            continue
        elif current_layer.name == 'Concat_input_C':
            d = current_layer([d, input_C])

        else:
            d = current_layer(d)
        
    fadein_model = base_models.Discriminator(img_shape=new_img_shape, 
                                      inputs=input_img, 
                                      outputs=d)

    return [straight_model, fadein_model]

