#! /usr/bin/env python

from pathlib import Path
from hypodisc.core.structures import GraphPattern

from rdf.formats import NTriples
from rdf.namespaces import XSD
from rdf.terms import IRIRef, Literal


REPO_IRI = IRIRef("https://gitlab.com/wxwilcke/hypodisc#")

def write_query(f_out:NTriples, pattern:GraphPattern, num_patterns:int,
                base:IRIRef, prefix_map:dict[str, str]) -> int:
    num_patterns += 1

    pattern_iri = base + f"#Query_{num_patterns}"
    pSupport = REPO_IRI + "hasSupport"
    pLength = REPO_IRI + "hasLength"
    pWidth = REPO_IRI + "hasWidth"
    pDepth = REPO_IRI + "hasDepth"
    pDot = REPO_IRI + "hasDOTRepresentation"
    pPattern = REPO_IRI + "hasPattern"

    qpattern = Literal(pattern.as_query(prefix_map),
                       datatype = XSD + "string")
    qdotrep = Literal(pattern.as_dot(prefix_map),
                       datatype = XSD + "string")
    qsupport = Literal(str(pattern.support),
                       datatype = XSD+"nonNegativeInteger")
    qlength = Literal(str(len(pattern)),
                      datatype = XSD+"nonNegativeInteger")
    qwidth = Literal(str(pattern.width()),
                     datatype = XSD+"nonNegativeInteger")
    qdepth = Literal(str(pattern.depth()),
                       datatype = XSD+"nonNegativeInteger")
    
    f_out.write((pattern_iri, pPattern, qpattern))
    f_out.write((pattern_iri, pDot, qdotrep))
    f_out.write((pattern_iri, pSupport, qsupport))
    f_out.write((pattern_iri, pLength, qlength))
    f_out.write((pattern_iri, pWidth, qwidth))
    f_out.write((pattern_iri, pDepth, qdepth))

    return num_patterns

def mkfile(directory:str, basename:str, extension:str) -> Path:
    """ Return path to a new file. Adds numerical suffix if
        the file already exists.

    :param directory:
    :type directory: str
    :param basename:
    :type basename: str
    :param extension:
    :type extension: str
    :rtype: Path
    """
    if not extension.startswith('.'):
        extension = '.' + extension

    out = Path(directory).joinpath(basename).with_suffix(extension)
    if not out.exists():
        return out

    suffix = 1
    while out.exists():
        outname = f"{basename}-{suffix}"
        out = Path(directory).joinpath(outname).with_suffix(extension)

        suffix += 1

    return out

class UnsupportedSerializationFormat(Exception):
    pass
