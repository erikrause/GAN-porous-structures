# GAN-porous-structures
Requirements:

    keras: 2.2.5
    tensorflow: 1.15.0 & 1.14.0 & 1.13.1
    plaidml(AMD GPU): 0.6.4

Use main to start training. After each sample_interval models_weights, logs and samples will be saved in DIRECTORY folder.

If DIRECTORY contains logs and weights, model will continue training from the last checkpoint. 

Logs files contains pickle objects with lists of logs.

# Examples

Dataset source: https://www.imperial.ac.uk/earth-science/research/research-groups/perm/research/pore-scale-modelling/micro-ct-images-and-networks/berea-sandstone/

Real(left)/fake(right) comprasion (128x128x128):

![real/fake](https://github.com/erikrause/GAN-porous-structures/blob/master/examples/real-fake.png)

Generated voxel array with 30% of porosity (rendered with marching cubes, 64x64x64):

<img src="https://github.com/erikrause/GAN-porous-structures/blob/master/examples/14.png" alt="marching cubes" width="50%"/>

Porosity interpolation from 52% to 13% (64x64x64):

<img src="https://github.com/erikrause/GAN-porous-structures/blob/master/examples/from%2040%25%20to%2087%25.gif" alt="marching cubes" width="50%"/>

[Docker web service](https://hub.docker.com/repository/docker/erikrause/porous_generator)
