# Hypodisc

Discovering Novel and Potentially Interesting Substructures in Knowledge Graphs

## Description

Hypodisc aims to discover novel and potentially interesting substructures in multimodal heterogeneous knowledge bases, encoded as [RDF](https://www.w3.org/TR/rdf12-concepts) knowledge graphs. Scholars can then use these substructures as a starting point to form new research hypotheses or to support existing ones, or to simply gain more insight into the knowledge, information, and data that is contained in their collections.

NB: this is a work in process.

### Multimodal Hypotheses

Hypodisc supports multimodal information of various types, which, in accordance to the RDF data model, are stored as string literals with an accompanying datatype or language tag. At present, Hypodisc understands numerical and temporal datatypes, as well as natural language and other strings. Literals with language tags are also treated as strings. To discover meaningful patterns in these data, a cluster-based approach is applied to all elements of the same datatype with a certain context. 

The full list of supported datatypes can be found below.


## Installation

To install this tool you will need the [git version control system](https://git-scm.com) and a recent [Python](https://www.python.org) setup which include `pip`. 

1) Clone this repository using `git`:

    git clone https://gitlab.com/wxwilcke/hypodisc.git

2) Change directory to the root of the tool:

    cd hypodisc/

3) Install the prerequisites (numpy, sklearn, and pyRDF) and the tool itself using pip:

    pip install .

Hypodisc is now installed and ready to use.

## Usage 

    usage: HypoDisc [-h] -d DEPTH -s MIN_SUPPORT [-o OUTPUT] [--max_size MAX_SIZE] [--max_width MAX_WIDTH] [--mode {A,T,AT,TA}] [--multimodal] [--p_explore P_EXPLORE] [--p_extend P_EXTEND] [--dry_run] [--seed SEED] [--verbose] [--version] input [input ...]
    
    positional arguments:
      input                 One or more knowledge graphs in (gzipped) NTriple or NQuad serialization format.
    
    options:
      -h, --help            
                            show this help message and exit
      -d DEPTH, --depth DEPTH
                            Depths to explore. Takes a range 'from:to', or a shorthand ':to' or 'to' if all depths up to that point are to be considered.
      -s MIN_SUPPORT, --min_support MIN_SUPPORT
                            Minimal pattern support.
      -o OUTPUT, --output OUTPUT
                            Path to write output to (defaults to current directory)
      --max_size MAX_SIZE   
                            Maximum context size (defaults to inf.)
      --max_width MAX_WIDTH
                            Maximum width of shell (defaults to inf.)
      --mode {A,T,AT}    
                            A[box], T[box], or both as candidates to be included in the pattern (defaults to AT)
      --multimodal          
                            Enable multimodal support (defaults to True)
      --namespace NAMESPACE
                            Add a custom prefix:namespace pair to be used in the output. This parameter can be used more than once to provide multiple mappings. Must be provided as 'prefix:namespace', eg 'ex:http://example.org/'.

      --p_explore P_EXPLORE
                            Probability of exploring candidate endpoint (defaults to 1.0)
      --p_extend P_EXTEND   
                            Probability of extending at candidate endpoint (defaults to 1.0)
      --dry_run             
                            Dry run without saving results (defaults to False)
      --seed SEED           
                            Set the seed for the random number generator (default to current time)
      --verbose, -v         
                            Print debug messages and warnings
      --version             
                            show program's version number and exit

## Walkthrough

For any RDF knowledge graph in N-Triple or N-Quad format, run the tool using:

    python hypodisc/run.py --depth <DEPTH> --min_support <SUPPORT>  [<KNOWLEDGE_GRAPH>, ...]  

See the `test/` directory for an example.

## Supported datatypes

The following datatypes are supported by Hypodisc:

Numbers:

```
- xsd:decimal
- xsd:double
- xsd:float
- xsd:integer
- xsd:long
- xsd:int
- xsd:short
- xsd:byte

- xsd:nonNegativeInteger
- xsd:nonPositiveInteger
- xsd:negativeInteger
- xsd:positiveInteger

- xsd:unsignedLong
- xsd:unsignedInt
- xsd:unsignedShort
- xsd:unsignedByte
```

Time/date:

```
- xsd:date
- xsd:dateTime
- xsd:dateTimeStamp
- xsd:gYear
- xsd:gYearMonth
- xsd:gMonthDay
- xsd:gMonth
- xsd:gDay
```

Strings:

```
- xsd:string
- xsd:normalizedString
- xsd:token
- xsd:language
- xsd:Name
- xsd:NCName
- xsd:ENTITY
- xsd:ID
- xsd:IDREF
- xsd:NMTOKEN
- xsd:anyURI
```


## Acknowledgements

The development of this tool is funded by the [CLARIAH](https://www.clariah.nl) project.
