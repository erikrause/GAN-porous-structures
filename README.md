# GAN-porous-structures
Versions:

    keras: 2.2.5
    tensorflow: 1.15.0 & 1.14.0 & 1.13.1
    plaidml(AMD GPU): 0.6.4

Use main to start training. After each sample_interval models_weights, logs and samples will be saved in DIRECTORY folder.

If DIRECTORY contains logs and weights, model will continue training from the last checkpoint. 
Logs files contains pickle objects with lists of logs.
