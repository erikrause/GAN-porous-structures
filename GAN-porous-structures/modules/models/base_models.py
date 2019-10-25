from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Embedding, Flatten, Input, Reshape, Dropout,
                          Concatenate, Layer)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend
import tensorflow as tf

from keras.initializers import RandomNormal
from keras.constraints import Constraint
#from keras.constraints import MinMaxNorm

from keras.layers.merge import _Merge

global constraint
global clip_value
global opt
global weight_init
global lr

constraint = None
clip_value = 0.01
current_backend = backend.backend()

lr = 0.001
opt = RMSprop(lr=lr)    #from vanilla WGAN paper
#opt = Adam(lr=0.001)        # from Progressive growing GAN paper
weight_init = RandomNormal(stddev=0.02)

# clip model weights to a given hypercube (vanilla WGAN)
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

if current_backend == 'tensorflow':
    constraint = ClipConstraint(clip_value)      # WGAN constraint for critics kernel_constraint.
    print(current_backend)
elif current_backend == 'plaidml':
    print(current_backend)              # With plaidml constrainsts will do in train loop.
else:
    print(' WARNING: BACKEND ' + current_backend + ' NOT SUPPORTED')


def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

class Generator(Model):
    def __init__(self, inputs, start_img_shape, outputs = None):

        self.start_img_shape = start_img_shape
        self.dims = len(self.start_img_shape) - 1

        if self.dims == 3:
            self.conv = Conv3DTranspose
            self.upsample = UpSampling3D
        elif self.dims == 2:
            self.conv = Conv2DTranspose
            self.upsample = UpSampling2D

        if outputs == None:
            model = self.__build(inputs)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

    def __build(self, z_dim):
  
        input_Z = Input(shape=(z_dim,))
        #input_C = Input(shape=(1,))

        #combined = Concatenate()([input_Z, input_C])

        units = 32
        channels = self.start_img_shape[-1]   #?
        hidden_shape = tuple(x//(2*2) for x in (self.start_img_shape[:-1]))
        for i in range(self.dims):
            units = units * hidden_shape[i]
        unints = units * channels   # channles не используется!

        hidden_shape = list(hidden_shape)
        hidden_shape.append(32)
        hidden_shape = tuple(hidden_shape)

        g = Dense(units)(input_Z)
        g = Reshape(hidden_shape)(g)
  
        g = self.conv(32, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
        g = self.upsample()(g)

        g = self.conv(16, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        g = BatchNormalization()(g)
        g = ReLU()(g)
        g = self.upsample()(g)
    
        g = self.conv(1, kernel_size=3, strides=1, padding='same', kernel_initializer = weight_init)(g)
        img = Activation('tanh')(g)

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
    
        d = Conv2D(8, kernel_size=1, strides = 1, padding='same', name='concat_layer', kernel_initializer = weight_init)(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = Dropout(rate = self.droprate)(d)
        d = AveragePooling2D()(d)

        d = Conv2D(16, kernel_size=3, strides = 1, padding='same', kernel_initializer = weight_init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        d = Dropout(rate = self.droprate)(d)
        d = AveragePooling2D()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        d = Dense(64, kernel_initializer = weight_init)(d)
        d = BatchNormalization()(d)
        d = ReLU()(d)
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
        classification = discriminator([img, generator.inputs[1]])
    
        model = Model(generator.inputs, classification)

        return model

class WGAN(Model):
    def __init__(self, generator, discriminator):
        discriminator.trainable = False
        model = self.__build(generator, discriminator)
        Model.__init__(self, model.inputs, model.outputs)
        self.compile(loss=wasserstein_loss, 
                     optimizer=opt)

    def __build(self, generator, discriminator):

        #input_Z = Input(shape=(z_dim,)) 
        #input_C = Input(shape=(1,))

        img = generator(generator.inputs)#generator([input_Z, input_C])
    
        # Combined Generator -> Discriminator model
        classification = discriminator(img)
    
        model = Model(generator.inputs, classification)

        return model

class Critic(Model):
    def __init__(self, img_shape=None, inputs = None, outputs = None, alpha = 0.2, droprate = 0.2):

        self.dims = len(img_shape) - 1
        if self.dims == 3:
            self.conv = Conv3D
            self.pool = AveragePooling3D
        elif self.dims == 2:
            self.conv = Conv2D
            self.pool = AveragePooling2D

        if outputs == None:
            self.alpha = 0.2
            self.droprate = droprate
            model = self.__build(img_shape)
            Model.__init__(self, model.inputs, model.outputs)
        elif outputs != None:
            Model.__init__(self, inputs, outputs)

        self.compile(loss=wasserstein_loss,
                      optimizer=opt)

    def __build(self, img_shape:tuple):

        input_img = Input(shape = img_shape)
        #input_C = Input(shape=(1,), name='Input_C')
    
        #d = conv(32, kernel_size=1, strides = 1, padding='same', name='concat_layer')(input_img)
        #d = LeakyReLU(alpha = self.alpha)(d) 
        #d = AveragePooling2D()(d)
    
        d = self.conv(32, kernel_size=3, strides = 1, padding='same', name='concat', kernel_initializer = weight_init, kernel_constraint=constraint)(input_img)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        #d = Dropout(rate = self.droprate)(d)
        d = self.pool()(d)

        d = self.conv(64, kernel_size=3, strides = 1, padding='same', name = 'conv', kernel_initializer = weight_init, kernel_constraint=constraint)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha = self.alpha)(d)
        #d = Dropout(rate = self.droprate)(d)
        d = self.pool()(d)
    
        d = Flatten()(d)

        #combined = Concatenate(name='Concat_input_C')([d, input_C])    

        d = Dense(128, kernel_initializer = weight_init, name='dense', kernel_constraint=constraint)(d)
        d = BatchNormalization()(d)
        d = ReLU()(d)
        #d = Dropout(rate = self.droprate)(d)
    
        d = Dense(1, activation='linear', kernel_constraint=constraint)(d)      #нужен ли constraint для Dense?


        model = Model(inputs=input_img, outputs=d)
    
        return model

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)