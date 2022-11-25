# HypoDisc Node Features Test

This repository contains a small artificial and multimodal dataset to test the feature encoders of the HypoDisc. The graph connecting these features has been randomly generated, such that all information must come from the features.

## Getting Started

Since the model does not directly work on RDF data, the first step entails the creation of the necessary input files. This can be done by calling the *generateInput* script with the splits are input. The splits are expected to be in HDT format.

To generate the input files, run:

    python generateInput.py -ts test/multimodal/train.hdt -ws test/multimodal/test.hdt -vs test/multimodal/valid.hdt -o test/multimodal/data/

Next, start the training process by calling the *train* script with the directory housing the just-generated files as input:

    python train.py -i test/multimodal/data/

The above call will require the preprocessing of the input data on every new run. Alternatively, the *mkdataset* can be used to create a HDF5 file of the preprocessed data, and which can be used as input instead.

To generate the HDF5 dataset file, run:

    python mkdataset.py -i test/multimodal/data/ -o test/multimodal/data/

Finally, run the HypoDisc with this file as input:

    python train.py -i test/multimodal/data/dataset.h5

Please see the help functions (`--help`) of these scripts for more information and more options.
