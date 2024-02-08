#! /usr/bin/env python

from __future__ import annotations
import gzip
from typing import Optional, Set, Tuple, Union
from typing_extensions import Self
from uuid import uuid4

import numpy as np
import scipy.sparse as sp

from rdf import NTriples, NQuads
from rdf.terms import IRIRef, Literal
from rdf.namespaces import RDF, RDFS, SKOS, OWL, XSD
from rdf.formats import RDF_Serialization_Format


# string datatype
XSD_STRING = XSD + "string"

# default prefixes
DEFAULT_PREFIX_MAP = {OWL.value: 'owl',
                      RDF.value: 'rdf',
                      RDFS.value: 'rdfs',
                      SKOS.value: 'skos',
                      XSD.value: 'xsd',
                      "http://purl.org/dc/terms/": 'dct',
                      "http://www.w3.org/ns/prov#": 'prov'}


def ns2pf(prefix_map: dict[str, str], iri: IRIRef) -> str:
    ns, name = irisplit(iri)
    if len(prefix_map) <= 0 or ns not in prefix_map.keys():
        return iri

    return f"{prefix_map[ns]}:{name}"


def mkprefixes(namespaces: Set[str],
               custom_prefix_map: Optional[dict[str, str]] = None)\
        -> dict[str, str]:
    prefix_map = dict()

    # add user-provided set (if any)
    if custom_prefix_map is not None:
        prefix_map.update(custom_prefix_map)

    # add other namespaces:
    i = 1
    for ns in namespaces:
        if ns in prefix_map.keys():
            continue

        if ns in DEFAULT_PREFIX_MAP.keys():
            # use common name
            prefix_map[ns] = DEFAULT_PREFIX_MAP[ns]

            continue

        prefix_map[ns] = f'ns{i}'
        i += 1

    return prefix_map


def irisplit(e: IRIRef) -> Tuple[str, str]:
    i = -1
    for i in range(len(e.value) - 1, 0, -1):
        if e.value[i] in ('/', '#'):
            break

    return e.value[:i+1], e.value[i+1:]


class UniqueLiteral(Literal):
    def __init__(self, value: str, datatype: Union[IRIRef, None] = None,
                 language: Union[str, None] = None) -> None:
        super().__init__(value=value,
                         datatype=datatype,
                         language=language)

        self._uuid = uuid4().hex

    def __eq__(self, other: UniqueLiteral) -> bool:
        return self._uuid == other._uuid

    def __hash__(self) -> int:
        return hash(self._uuid)


class KnowledgeGraph():
    """ Knowledge Graph stored in vector representation plus query functions
    """

    def __init__(self, rng: np.random.Generator,
                 paths: list[str]) -> None:
        """ Knowledge Graph stored in vector representation plus query
            functions

        :param rng:
        :type rng: np.random.Generator
        :rtype: None
        """
        self._rng = rng
        self.paths = paths
        self.namespaces = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def parse(self) -> None:
        """ Parse graph on file level.

        Supports plain or gzipped NTriple or NQuad files

        :param path:
        :type path: list[str]
        :rtype: None
        """
        nodes = dict()  # type: dict[Union[IRIRef,Literal], int]
        relations = dict()  # type: dict[IRIRef, int]
        annotations = dict()  # type: dict[Union[IRIRef, Literal], int]
        literals_attr = dict()  # type: dict[int, int]
        facts = list()  # type: list[list[list[int]]]
        namespaces = set()  # type: set[str]

        n_idx, r_idx, d_idx = 0, 0, 0

        annotations[XSD_STRING] = 0
        d_idx += 1
        for path in self.paths:
            parts = path.split('.')
            is_gzipped = parts[-1] == "gz"
            suffix = parts[-1] if not is_gzipped else parts[-2]

            parser = None
            if suffix == "nt":
                parser = NTriples
            elif suffix == "nq":
                parser = NQuads
            else:
                raise Exception("Supports graphs in NTriples or NQuads format."
                                f" Unsupported format: {suffix}")

            if is_gzipped:
                with gzip.open(path, mode='r') as gf:
                    with parser(data=gf.read(), mode='r') as g:
                        n_idx, r_idx, d_idx, namespaces \
                                = self._parse(g, (n_idx, r_idx, d_idx),
                                              (nodes, relations,
                                               annotations, literals_attr,
                                               facts))
            else:
                with parser(path=path, mode='r') as g:
                    n_idx, r_idx, d_idx, namespaces\
                            = self._parse(g, (n_idx, r_idx, d_idx),
                                          (nodes, relations,
                                           annotations, literals_attr,
                                           facts))

        self.namespaces = namespaces
        self._parse_vectorize(facts, nodes, relations,
                              annotations, literals_attr)

    def _parse(self, g: RDF_Serialization_Format,
               counters: tuple[int, int, int],
               data: tuple[dict, dict, dict, dict, list[list[list[int]]]])\
            -> tuple[int, int, int, set[str]]:
        """ Parse content of graph

        Generate indices for nodes, relations, and facts
        Optimize on memory use by streaming the source graph

        :param g:
        :type g: RDF_Serialization_Format
        :param counters:
        :type counters: tuple[int, int, int]
        :param data:
        :type data: tuple[dict, dict, dict, list[list[list[int]]]]
        :rtype: tuple[int, int, set[str]]
        """
        n_idx, r_idx, d_idx = counters
        nodes, relations, annotations, literals_attr, facts = data

        namespaces = set()
        for s, p, o in g.parse():
            for e in [s, p, o]:
                if type(e) is not IRIRef:
                    continue

                ns, _ = irisplit(e)
                namespaces.add(ns)

            # assign indices to elements
            if s in nodes.keys():
                s_idx = nodes[s]
            else:
                s_idx = n_idx
                nodes[s] = s_idx

                n_idx += 1

            if p in relations.keys():
                p_idx = relations[p]
            else:
                p_idx = r_idx
                relations[p] = p_idx

                facts.append([[], []])

                r_idx += 1

            if isinstance(o, Literal):
                o = UniqueLiteral(o.value,
                                  o.datatype,
                                  o.language)

                if o.language is not None\
                        and o.language not in annotations.keys():
                    annotations[o.language] = d_idx

                    d_idx += 1
                elif o.datatype is not None\
                        and o.datatype not in annotations.keys():
                    annotations[o.datatype] = d_idx

                    d_idx += 1

            if o in nodes.keys():
                o_idx = nodes[o]
            else:
                o_idx = n_idx
                nodes[o] = o_idx

                n_idx += 1

            # store s,p,o as indices
            facts[p_idx][0].append(s_idx)
            facts[p_idx][1].append(o_idx)

            # save datatype or language tag
            if isinstance(o, UniqueLiteral):
                if o.language is not None\
                        and o.language in annotations.keys():
                    literals_attr[o_idx] = annotations[o.language]
                elif o.datatype is not None\
                        and o.datatype in annotations.keys():
                    literals_attr[o_idx] = annotations[o.datatype]
                else:
                    # default to string
                    literals_attr[o_idx] = annotations[XSD_STRING]

        for dt in set(annotations.keys()):
            if type(dt) is not IRIRef:
                continue

            ns, _ = irisplit(dt)
            namespaces.add(ns)

        return n_idx, r_idx, d_idx, namespaces

    def _parse_vectorize(self, facts: list[list[list[int]]],
                         nodes: dict[Union[IRIRef, Literal], int],
                         relations: dict[IRIRef, int],
                         annotations: dict[Union[IRIRef, Literal], int],
                         literals_attr: dict[int, int]) -> None:
        """ Vectorize graph representation.

        :param facts:
        :type facts: list[list[list[int]]]
        :param nodes:
        :type nodes: dict[Union[IRIRef, Literal], int]
        :param relations:
        :type relations: dict[IRIRef, int]
        :param datatypes:
        :type datatypes: dict[int, Union[IRIRef, Literal]]
        :rtype: None
        """
        # statistics
        self.num_facts = 0
        self.num_nodes = len(nodes)
        self.num_relations = len(relations)

        self.A = list()
        for p_idx in sorted(relations.values()):
            n = len(facts[p_idx][0])
            data = np.ones(n, dtype=bool)
            self.A.append(sp.coo_array((data, (facts[p_idx][0],
                                               facts[p_idx][1])),
                                       shape=(self.num_nodes,
                                              self.num_nodes),
                                       dtype=bool))

            self.num_facts += n

        # lookup and reverse lookup tables
        self.n2i = nodes
        self.i2n = np.array(list(self.n2i.keys()))

        self.r2i = relations
        self.i2r = np.array(list(self.r2i.keys()))

        self.a2i = annotations
        self.i2a = np.array(list(self.a2i.keys()))

        self.ni2ai = literals_attr
