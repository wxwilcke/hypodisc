#! /usr/bin/env python

import gzip
from typing import Optional, List, Union

import numpy as np

from rdf import NTriples, NQuads
from rdf.terms import IRIRef, Literal, BNode
from rdf.namespaces import RDF, XSD
from rdf.formats import RDF_Serialization_Format


class KnowledgeGraph():
    """ Knowledge Graph stored in vector representation plus query functions
    """

    def __init__(self, rng:np.random.Generator,
                 path:Optional[str] = None) -> None:
        self._rng = rng

        if path is not None:
            self.parse(path)

    def parse(self, path:str) -> None:
        """ Parse graph on file level.

        Supports plain or gzipped NTriple or NQuad files

        :param path:
        :type path: str
        :rtype: None
        """
        path_parts = path.split('.')
        is_gzipped = True if path_parts[-1] == "gz" else False
        ext = path_parts[-2] if is_gzipped else path_parts[-1]

        parser = None
        if ext == "nt":
            parser = NTriples
        elif ext == "nq":
            parser = NQuads
        else:
            raise Exception("Supports graphs in NTriples or NQuads format."\
                            f" Unsupported format: {ext}")

        if is_gzipped:
            with gzip.open(path, mode='r') as gf:
                with parser(data=gf.read(), mode='r') as g:
                    self._parse(g)
        else:
            with parser(path=path, mode='r') as g:
                self._parse(g)

    def _parse(self, g:RDF_Serialization_Format) -> None:
        """ Parse content of graph

        Generate indices for nodes, relations, and facts
        Optimize on memory use by streaming the source graph

        :param g:
        :type g: RDF_Serialization_Format
        :rtype: None
        """
        nodes = dict()  # entities, blank nodes, and literals
        relations = dict()  # predicates
        datatypes = dict()  # datatype or language tag
        facts = (list(), list(), list())
        
        # string datatype
        xsd_string = XSD + "string"

        n_idx, r_idx = 0, 0
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
            if isinstance(o, Literal):
                if o.language is not None:
                    datatypes[o_idx] = o.language
                elif o.datatype is not None:
                    datatypes[o_idx] = o.datatype
                else:
                    # default to string
                    datatypes[o_idx] = xsd_string

        self._parse_vectorize(facts, nodes, relations, datatypes)

    def _parse_vectorize(self,
                         facts:tuple[List[IRIRef], List[IRIRef], List[IRIRef]],
                         nodes:dict[IRIRef, int], relations:dict[IRIRef, int],
                         datatypes:dict[int, Union[IRIRef, Literal]]) -> None:
        """_parse_vectorize.

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
