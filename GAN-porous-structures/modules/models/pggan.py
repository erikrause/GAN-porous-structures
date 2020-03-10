from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Multiply, Reshape, Dropout,
                          Concatenate, Layer, Add)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
from modules.models.layers import *
#import tensorflow as tf

#import tensorflow.python.keras.backend as K

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
                backend.set_value(layer.alpha, alpha)
    return alpha
def update_lod(g_model, d_model, step, n_steps, alpha = -1):
    # calculate current alpha (linear from 0 to 1)
    if alpha == -1:
        #current_alpha = backend.eval(layer.alpha)#K.get_value(layer.alpha)
        current_alpha = backend.get_value(d_model.cur_lod)
        remaining_alpha = 1 - current_alpha
        remaining_steps = n_steps - step
        if remaining_steps == 0:
            alpha = 1.0
        else:
            #alpha = step / float(n_steps - 1)
            dalpha = remaining_alpha / remaining_steps
            alpha = current_alpha + dalpha

        if hasattr(g_model, 'cur_lod'): 
                    backend.set_value(g_model.cur_lod,np.float32(alpha))
        if hasattr(d_model, 'cur_lod'): 
                    backend.set_value(d_model.cur_lod,np.float32(alpha))
    return alpha

def add_block(old_model, n_filters, filter_size):
    models = []

    if isinstance(old_model, base_models.Discriminator):
        models = __add_discriminator_block(old_model, n_filters, filter_size)

    elif isinstance(old_model, base_models.Generator):
        models = __add_generator_block(old_model, n_filters, filter_size)

    # If needs to build new gan from list[discriminator, generator]:
    elif isinstance(old_model, list):
        discriminators = old_model[0]
        generators = old_model[1]
        models = __add_wgan_block(discriminators, generators)

    return models

def __add_generator_block(old_model, n_filters, filter_size):
    # get the end of the last block
    block_end = old_model.layers[-3].output
    conv = old_model.conv
    upsample = old_model.upsample

    # upsample and define new block
    upsampling = upsample()(block_end)
    g = conv(n_filters, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=base_models.weight_init)(upsampling)
    g = BatchNormalization()(g)
    g = LeakyReLU(base_models.alpha)(g)

    #g = conv(n_filters, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=base_models.weight_init)(g)
    #g = BatchNormalization()(g)
    #g = LeakyReLU(base_models.alpha)(g)
    
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

    #out_image2 = upsample()(old_model.layers[-1].output)
    cur_lod = K.variable(np.float(0.0), dtype='float32', name='cur_lod')
    merged = LODSelectLayer(cur_lod, name='Glod')([out_image, old_model.layers[-1].output])
    ## define new output image as the weighted sum of the old and new models
    #merged = WeightedSum()([out_image2, out_image])        # может сввпнуть?

    # define model
    fadein_model = base_models.Generator(inputs=old_model.inputs,
                                         start_img_shape=old_model.start_img_shape,
                                         outputs=merged)
    fadein_model.cur_lod = cur_lod
    return [straight_model, fadein_model]

def __add_gan_block(discriminators, generators):
    fadein_model = base_models.GAN(generators[1], discriminators[1])
    straight_model = base_models.GAN(generators[0], discriminators[0])
    
    return [straight_model, fadein_model]

def __add_wgan_block(discriminators, generators):
    fadein_model = base_models.WGAN(generators[1], discriminators[1])
    straight_model = base_models.WGAN(generators[0], discriminators[0])
    
    return [straight_model, fadein_model]

def __add_discriminator_block(old_model, n_filters=64, filter_size=3, n_input_layers=4):

    input_shape = list(old_model.get_input_shape_at(0))
    for i in range(1,4):
        input_shape[i] = input_shape[i]*2

    new_img_shape = input_shape[1:]

    input_img = Input(batch_shape=input_shape)    # was new_img_shape

    conv = old_model.conv
    pool = old_model.pool
    
    # debug
    #print(n_filters)
    #print(old_model.summary())
    # New block/
    d = conv(n_filters//2, 
               kernel_size=filter_size, 
               strides=1, 
               padding='same', 
               kernel_initializer=base_models.weight_init)(input_img)
    d = BatchNormalization()(d)
    d = LeakyReLU(base_models.alpha)(d)

    #n_filters_last = old_model.layers[1].filters  #количество старых фильтров входа
    #kernel_size_last = old_model.layers[1].kernel_size

    #d = conv(n_filters, 
    #           kernel_size=filter_size, 
    #           strides=1, 
    #           padding='same', 
    #           kernel_initializer=base_models.weight_init)(d)
    #d = BatchNormalization()(d)
    #d = LeakyReLU(base_models.alpha)(d)
    
    d = conv(n_filters,
               kernel_size=filter_size, 
               strides=1, padding='same', 
               kernel_initializer=base_models.weight_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(base_models.alpha)(d)
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

        # debug
        prob = current_layer.get_weights()
        
    straight_model = base_models.Discriminator(img_shape=new_img_shape,
                                        inputs=input_img, 
                                        outputs=d)

    downsample = pool()(input_img)
    
    block_old = downsample
    for i in range(1, n_input_layers):
        block_old = old_model.layers[i](block_old)
    
    #d = WeightedSum()([block_old, block_new])
    cur_lod = K.variable(np.float32(0.0), dtype='float32', name='cur_lod')
    d = LODSelectLayer(cur_lod, name='Glod')([block_new, block_old])
    
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

    fadein_model.cur_lod = cur_lod

    print("fadein:")
    print(fadein_model.summary())
    print("straight:")
    print(straight_model.summary())

    return [straight_model, fadein_model]

'''
shape = (16,16,16,1)
# ЗАГЛУШКИ!!!
def __add_discriminator_block(old_model, n_filters=64, filter_size=3, n_input_layers=4,):
    model = base_models.Discriminator(shape)
    return [model, model]

def __add_generator_block(old_model, n_filters=64, filter_size=3):
    model = base_models.Generator(70, shape)
    return [model, model]
'''