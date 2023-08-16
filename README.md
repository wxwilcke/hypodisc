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

## Walkthrough

For any RDF knowledge graph in N-Triple or N-Quad format, run the tool using:

    python hypodisc/run.py --input <KNOWLEDGE_GRAPH> --depth <DEPTH> --min_support <SUPPORT> --min_confidence <CONFIDENCE>

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
