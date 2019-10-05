from modules.models import PGGAN, base_models

class ModelStrogae(object):
    def __init__(self):
        self.discriminators = []
        self.generators = []
        self.gans = []

    def buildModels(self, start_shape, z_dim, n_blocks, n_filters, filter_sizes):
        base_discriminator = base_models.Discriminator(start_shape)
        self.discriminators.append([base_discriminator, base_discriminator])   

        base_generator = base_models.Generator(z_dim)
        self.generators.append([base_generator, base_generator])

        for i in range(1, n_blocks):
            #Block for discriminator/
            old_discriminator = self.discriminators[i - 1][0]
	        # create new model for next resolution
            new_discriminators = PGGAN.add_discriminator_block(old_discriminator,
                                                               n_filters = n_filters[i]//8,
                                                               filter_size = filter_sizes[i])
            self.discriminators.append(new_discriminators)
            #/Block for discriminator

            #Block for generator/
            old_generator = self.generators[i - 1][0]
            new_generators = PGGAN.add_generator_block(old_generator,
                                                       n_filters = n_filters[i],
                                                       filter_size = filter_sizes[i])
            self.generators.append(new_generators)
            #/Block for generator

        self.gans = PGGAN.build_composite(self.discriminators, self.generators)
