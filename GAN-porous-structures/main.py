#################
#FOR AMD DEVICES:
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#################

import models

n_blocks = 4

n_filters = {1: 64,
             2: 32,
             3: 16}

filter_size = {1: (3,3),
               2: (3,3),
               3: (3,3)}        ## Протестировать фильтры 5х5

discriminators = []
generators = []
# Build and compile the Discriminator
base_discriminator = models.build_base_discriminator(img_shape)

discriminator = models.build_discriminator(base_discriminator)
discriminator.trainable = False
discriminators.append([discriminator, discriminator])   

# Build the Generator
base_generator = models.build_base_generator(z_dim+1)
generator = models.build_generator(z_dim, base_generator)
generators.append([generator, generator])
