# Deep Hypothesis Discovery on RDF Knowledge Graphs (HypoDisc)

This tool aims at discovering novel and potentially interesting hypotheses in RDF knowledge graphs.

The goal of discovering hypotheses is redefined as a multi-hop link prediction and embedding clustering problem. First, a multi-hop link prediction model is trained on a subset of the graph, while simultaneously encouraging the embeddings of entities whose context is similar to cluster together in the embedding space. The second step takes the centroids of the clusters, and uses them as _representatives_ of these clusters with which to rank the set of paths that exist in the cluster, thereby swapping the original embedding by the new representative. The top-k paths form the hypotheses for that cluster.

**This is a work in progress**

## Acknowledgements

This research is partly funded by [CLARIAH](https://www.clariah.nl)
