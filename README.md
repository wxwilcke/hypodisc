# HypoDisc

Discover Novel and Potentially Interesting Substructures in Multimodal Knowledge Graphs

## Description

NB: this is a work in process.

## Installation

1) Clone this repository:

    git clone https://gitlab.com/wxwilcke/hypodisc.git

2) Change directory to the root of the tool:

    cd hypodisc/

3) Install the prerequisites and the tool itself using pip:

    pip install .

## Walkthrough

For any RDF knowledge graph in N-Triple or N-Quad format, run the tool using:

    python hypodisc/run.py --input <KNOWLEDGE_GRAPH> --depth <DEPTH> --min_support <SUPPORT> --min_confidence <CONFIDENCE>

See the `test/` directory for an example.
