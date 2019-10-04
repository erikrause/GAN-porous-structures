#################
#FOR AMD DEVICES:
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#################




from modules import PGGAN, base_models

from modules.DataLoader import DataLoader


img_rows = 16
img_cols = 16
channels = 1
droprate = 0.2

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100

#DIRECTORY = '/content/drive/My Drive/GAN/PGGANv5/beadpack'
#DATASET_DIR = '/content/drive/My Drive/GAN/datasets/beadpack'

DIRECTORY = 'E:/Практика/beadpack'
DATASET_DIR = 'E:/Практика/beadpack/dataset'

data_loader = DataLoader(DATASET_DIR+'/{}.png', 500, (500,500))

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
base_discriminator = base_models.Discriminator(img_shape)

#discriminator = base_models.add_input_c_to_discriminator(base_discriminator)
#base_discriminator.trainable = False
discriminators.append([base_discriminator, base_discriminator])   

#base_generator = base_models.build_generator(z_dim+1)
base_generator = base_models.Generator(z_dim)
#generator = models.build_generator(z_dim, base_generator)
generators.append([base_generator, base_generator])

for i in range(1, n_blocks):
    #Block for discriminator/
    old_discriminator = discriminators[i - 1][0]
	# create new model for next resolution
    new_discriminators = PGGAN.add_discriminator_block(old_discriminator,
                                                       n_filters = n_filters[i]//8,
                                                       filter_size = filter_size[i])
    discriminators.append(new_discriminators)
    #/Block for discriminator

    #Block for generator/
    old_generator = generators[i - 1][0]
    new_generators = PGGAN.add_generator_block(old_generator,
                                               n_filters = n_filters[i],
                                               filter_size = filter_size[i])
  
    generators.append(new_generators)
    #/Block for generator

gans = PGGAN.build_composite(discriminators, generators)

generators[n_blocks-1][0].summary()

