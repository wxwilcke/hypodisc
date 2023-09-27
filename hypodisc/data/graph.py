#! /usr/bin/env python

from __future__ import annotations
import gzip
from pathlib import Path
from typing import Optional, List, Union
from uuid import uuid4

import numpy as np

from rdf import NTriples, NQuads
from rdf.terms import IRIRef, Literal, BNode
from rdf.namespaces import RDF, XSD
from rdf.formats import RDF_Serialization_Format


# string datatype
XSD_STRING = XSD + "string"

class UniqueLiteral(Literal):
    def __init__(self, value:str, datatype:Union[IRIRef,None] = None,
                 language:Union[str,None] = None) -> None:
        super().__init__(value = value,
                         datatype = datatype,
                         language = language)

        self._uuid = uuid4().hex

    def __eq__(self, other:UniqueLiteral) -> bool:
        return self._uuid == other._uuid

    def __hash__(self) -> int:
        return hash(self._uuid)

class KnowledgeGraph():
    """ Knowledge Graph stored in vector representation plus query functions
    """

    def __init__(self, rng:np.random.Generator) -> None:
        """ Knowledge Graph stored in vector representation plus query
            functions

        :param rng:
        :type rng: np.random.Generator
        :rtype: None
        """
        self._rng = rng

    def parse(self, paths:list[str]) -> None:
        """ Parse graph on file level.

        Supports plain or gzipped NTriple or NQuad files

        :param path:
        :type path: list[str]
        :rtype: None
        """
        nodes = dict()  # entities, blank nodes, and literals
        relations = dict()  # predicates
        datatypes = dict()  # datatype or language tag
        facts = (list(), list(), list())  # graph by index

        n_idx, r_idx = 0, 0
        for path in paths:
            parts = path.split('.')
            is_gzipped = parts[-1] == ".gz"
            suffix = parts[-1] if not is_gzipped else parts[-2]

            parser = None
            if suffix == "nt":
                parser = NTriples
            elif suffix == "nq":
                parser = NQuads
            else:
                raise Exception("Supports graphs in NTriples or NQuads format."\
                                f" Unsupported format: {suffix}")

            if is_gzipped:
                with gzip.open(path, mode='r') as gf:
                    with parser(data=gf.read(), mode='r') as g:
                        n_idx, r_idx = self._parse(g, (n_idx, r_idx),
                                                  (nodes, relations,
                                                   datatypes, facts))
            else:
                with parser(path=path, mode='r') as g:
                    n_idx, r_idx = self._parse(g, (n_idx, r_idx),
                                              (nodes, relations,
                                               datatypes, facts))


        self._parse_vectorize(facts, nodes, relations, datatypes)

    def _parse(self, g:RDF_Serialization_Format, counters:tuple[int, int],
               data:tuple[dict, dict, dict,
                          tuple[list[IRIRef], list[IRIRef], list[IRIRef]]])\
                       -> tuple[int, int]:
        """ Parse content of graph

        Generate indices for nodes, relations, and facts
        Optimize on memory use by streaming the source graph

        :param g:
        :type g: RDF_Serialization_Format
        :rtype: None
        """
        n_idx, r_idx = counters
        nodes, relations, datatypes, facts = data
        for s, p, o in g.parse():
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

                r_idx += 1

            if isinstance(o, Literal):
                o = UniqueLiteral(o.value,
                                  o.datatype,
                                  o.language)

            if o in nodes.keys():
                o_idx = nodes[o]
            else:
                o_idx = n_idx
                nodes[o] = o_idx

                n_idx += 1

            # store s,p,o as indices
            facts[0].append(p_idx)            
            facts[1].append(s_idx)            
            facts[2].append(o_idx)            
 
            # save datatype or language tag
            if isinstance(o, UniqueLiteral):
                if o.language is not None:
                    datatypes[o_idx] = o.language
                elif o.datatype is not None:
                    datatypes[o_idx] = o.datatype
                else:
                    # default to string
                    datatypes[o_idx] = XSD_STRING

        return n_idx, r_idx

    def _parse_vectorize(self,
                         facts:tuple[list[IRIRef], list[IRIRef], list[IRIRef]],
                         nodes:dict[IRIRef, int], relations:dict[IRIRef, int],
                         datatypes:dict[int, Union[IRIRef, Literal]]) -> None:
        """ Vectorize graph representation.

        :param facts:
        :type facts: tuple[List[IRIRef], List[IRIRef], List[IRIRef]]
        :param nodes:
        :type nodes: dict[IRIRef, int]
        :param relations:
        :type relations: dict[IRIRef, int]
        :param datatypes:
        :type datatypes: dict[int, Union[IRIRef, Literal]]
        :rtype: None
        """
        # statistics
        self.num_nodes = len(nodes)
        self.num_relations = len(relations)

        self.A = np.zeros((self.num_relations, self.num_nodes, self.num_nodes),
                          dtype=bool)
        self.A[facts] = True

        # lookup and reverse lookup tables
        self.n2i = nodes
        self.i2n = np.array(list(self.n2i.keys()))

        self.r2i = relations
        self.i2r = np.array(list(self.r2i.keys()))

        self.i2d = datatypes


