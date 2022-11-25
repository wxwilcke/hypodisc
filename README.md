# Deep Hypothesis Discovery on RDF Knowledge Graphs (HypoDisc)

This tool aims at discovering novel and potentially interesting hypotheses in RDF knowledge graphs.

The goal of discovering hypotheses is redefined as a multi-hop link prediction and embedding clustering problem. First, a multi-hop link prediction model is trained on a subset of the graph, while simultaneously encouraging the embeddings of entities whose context is similar to cluster together in the embedding space. The second step takes the centroids of the clusters, and uses them as _representatives_ of these clusters with which to rank the set of paths that exist in the cluster, thereby swapping the original embedding by the new representative. The top-k paths form the hypotheses for that cluster.

**This is a work in progress**

## Getting Started

1) To install, clone the repository and run:

    cd hypodisc/ && pip install .

This will install this package on your computer and will take care of any necessary dependencies.

2) Since the model does not directly work on RDF data, the first step entails the creation of the necessary input files. This can be done by calling the *generateInput* script with the training (-ts), testing (-ws), and, optionally, validation (-vs) splits are input. The splits are expected to be RDF graphs in [HDT format](https://www.rdfhdt.org). A tool to convert many common RDF serialization formats to HDT can be found [here](https://github.com/rdfhdt/hdt-cpp).

To generate the input files, run:

    python generateInput.py -ts train.hdt -ws test.hdt -vs valid.hdt -o myproject/

This will take files *train.hdt*, *test.hdt*, and *valid.hdt*, and will convert their content to the [KGbench](http://kgbench.info) format, which can more easily be processed by a data science pipeline. The output is written to the specified directory (-o), here called *myproject*.

3) Next, start the training process by calling the *train* script with the directory housing the just-generated files as input (-i):

    python train.py -i myproject/

This will pre-process the data and use these data to train the model by learning clusters of entities with similar characteristics. To avoid having to pre-process the data each time, the *mkdataset* can be used to create a HDF5 file of the preprocessed data which can be used as input instead:

(optional) To generate the HDF5 dataset file, run:

    python mkdataset.py -i myproject/ -o myproject/

which will create a file called *dataset.h5* in the specified directory (-o).

Finally, run the HypoDisc with this file as input:

    python train.py -i myproject/dataset.h5

Please see the help functions (`--help`) of these scripts for more information and more options.

## Multimodal learning

This pipeline includes experimental support for multimodal features, stored as literal nodes. These are disabled by default, but can be enabled by specifying one or more of the following modalities: numerical, temporal, textual, spatial, and visual. These modalities map to their corresponding [XSD datatypes](https://www.w3.org/TR/xmlschema11-2/#built-in-datatypes), or [b64Image](http://kgbench.info) for images encoded as string literals. Note that, for literals to be considered a member of a modality they should correctly be annotated with a datatype or language tag..

To create a HDF5 file which in includes text and images, run:

    python mkdataset.py -i myproject/ -o myproject/ -m textual visual

To now include these modalities in the learning process, repeat the same when calling the *train* module:

    python train.py -i myproject/dataset.h5 -m textual visual

Note that, once the modalities are included in the HDF5 file, they can be enabled or disabled on the fly during training without regenerating the HDF5 file.

## Generating Clusters

[t-SNE](https://lvdmaaten.github.io/tsne/) can be used to visualize the clusters in the entity embedding space. To enable this feature, you first need to install t-SNE as a submodule of this repository, which can be done using the following commands:

    git submodule init && git submodule update
    cd bhtsne/ && g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O3 && cd -

Once installed, you can generate the clusters each evaluation by starting the training sequence with the `--save_clusters` flag. This will write the clusters to the output directory (e.g. _myproject_). The `plotClusters` helper script can then be used to visualize them:

    python plotClusters myproject/

## Acknowledgements

This research is partly funded by [CLARIAH](https://www.clariah.nl)
