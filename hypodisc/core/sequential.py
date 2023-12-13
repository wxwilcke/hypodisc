#! /usr/bin/env python

from random import random
from time import time
from multiprocessing import Manager
from typing import Counter, Literal, Optional, Union

import numpy as np
from hypodisc.data.utils import write_query

from rdf.formats import NTriples
from rdf.namespaces import RDF, RDFS, XSD
from rdf.terms import IRIRef
from rdf.terms import Literal as rdfLiteral

from hypodisc.core.utils import predict_hash
from hypodisc.data.graph import KnowledgeGraph
from hypodisc.core.structures import (Assertion, GraphPattern,
                                      ResourceWrapper, TypeVariable,
                                      DataTypeVariable,
                                      Variable,
                                      MultiModalNumericVariable,
                                      MultiModalStringVariable,
                                      ObjectTypeVariable)
from hypodisc.multimodal.clustering import (compute_clusters,
                                            SUPPORTED_XSD_TYPES)
from hypodisc.multimodal.datatypes import (XSD_DATEFRAG, XSD_DATETIME,
                                           XSD_NUMERIC)


IGNORE_PREDICATES = {RDF + 'type', RDFS + 'label'}


def generate(rng:np.random.Generator, kg:KnowledgeGraph,
             depths:range, min_support:int,
             p_explore:float, p_extend:float,
             max_length:int, max_width:int,
             multimodal:bool, out_writer:Optional[NTriples],
             out_prefix_map:Optional[dict[str,str]],
             out_ns:Optional[IRIRef],
             mode:Literal["A", "T", "AT"]) -> None: 
    """ Generate all patterns up to and including a maximum depth which
        satisfy a minimal support.

    :param kg:
    :type kg: KnowledgeGraph
    :param depths:
    :type depths: range
    :param min_support:
    :type min_support: int
    :param p_explore:
    :type p_explore: float
    :param p_extend:
    :type p_extend: float
    :param max_length:
    :type max_length: int
    :param max_width:
    :type max_width: int
    :param multimodal:
    :type multimodal: bool
    :param mode:
    :type mode: Literal["A", "AT", "T"]
    :rtype: None
    """

    t0 = time()
    root_patterns = init_root_patterns(rng, kg, min_support,
                                       mode, multimodal)

    parents = dict()
    npruned = 0
    num_patterns = 0
    for depth in range(0, depths.stop):
        print("exploring depth {} / {}".format(depth+1, depths.stop))

        if depth == 0:
            parents = root_patterns

        visited = set()
        derivatives = dict()
        for name in parents.keys():
            print(f" type {name}", end=" ")

            derivatives[name] = set()
            for pattern in parents[name]:
                if len(pattern) >= max_length or pattern.width() >= max_width:
                    continue
                    
                # Gather candidate endpoints at this depth.
                # Only consider unbound object type variables since we
                # cannot connect to literals or data type variables, and
                # bound entities already have a fixed context.
                if depth <= 0:
                    endpoints = {pattern.root}
                else:
                    endpoints = {a.rhs for a in pattern.distances[depth-1]
                                 if isinstance(a.rhs, ObjectTypeVariable)}

                candidates = set()
                for endpoint in endpoints:
                    if endpoint.value not in root_patterns.keys():
                        # no extension available
                        continue

                    if p_explore < random():
                        # skip this endpoint with probability p_explore
                        continue

                    # Gather all candidate extensions that can connect
                    # to an object type variable of the relevant type.
                    for base_pattern in root_patterns[endpoint.value]:                            
                        if p_extend < random():
                            # skip this extension with probability p_extend
                            continue

                        # get base assertion
                        extension = base_pattern.assertion

                        # prune                   
                        if pattern.contains_at_depth(extension, depth):
                            continue

                        pattern_hash = predict_hash(pattern,
                                                    endpoint,
                                                    extension)
                        if pattern_hash in visited:
                            continue
                        visited.add(pattern_hash)
                            
                        if "T" not in mode\
                                and isinstance(extension.rhs,
                                               TypeVariable):
                            # limit extensions to Abox
                            continue
                        if "A" not in mode\
                                and not isinstance(extension.rhs,
                                                   TypeVariable):
                            # limit extensions to Tbox
                            continue

                        candidates.add((endpoint, extension))


                    extensions = explore(pattern,
                                         candidates = candidates,
                                         max_length = max_length,
                                         max_width = max_width,
                                         min_support = min_support)

                    if out_writer is not None and out_prefix_map is not None:
                        for pattern in extensions:
                            num_patterns = write_query(out_writer, pattern,
                                                       num_patterns, out_ns,
                                                       out_prefix_map)

                    derivatives[name] |= extensions

            print("(+{} discovered)".format(len(derivatives[name])))

        # omit exhausted classes from next iteration
        parents = {k:v for k,v in derivatives.items()
                   if len(v) > 0}

    duration = time()-t0
    print('discovered {} patterns in {:0.3f}s'.format(num_patterns, duration),
          end="")

    if npruned > 0:
        print(" ({} pruned)".format(npruned))
    else:
        print()

def explore(parent:GraphPattern, candidates:set,
            max_length:int, max_width:int, min_support:int) -> set:
    """ Explore all predicate-object pairs which where added by the previous
    iteration as possible endpoints to expand from.

    :param pattern:
    :type pattern: GraphPattern
    :param candidates:
    :type candidates: set
    :param max_length:
    :type max_length: int
    :param max_width:
    :type max_width: int
    :param min_support:
    :type min_support: int
    :rtype: set
    """
    derivatives = set()  # valid patterns
    skipped = set()  # invalid patterns
    with Manager() as manager:
        qexplore = manager.Queue()
        qexplore.put(parent)
        
        while not qexplore.empty():
            pattern = qexplore.get()

            if len(pattern) >= max_length or pattern.width() >= max_width:
                continue

            for endpoint, extension in candidates:
                pattern_hash = predict_hash(pattern, endpoint, extension)
                if extension in pattern:
                    # extension is already part of pattern
                    continue
                elif pattern_hash in derivatives:
                    # already seen and added this pattern
                    continue
                elif pattern_hash in skipped:
                    # already seen but skipped this pattern
                    continue

                pattern_new = extend(pattern, endpoint, extension)

                if pattern_new.support > min_support:
                    qexplore.put(pattern_new)  # explore further extensions
                    derivatives.add(pattern_new)
                else:
                    skipped.add(pattern_hash)

    return derivatives

def extend(pattern:GraphPattern, endpoint:Variable, extension:Assertion)\
        -> GraphPattern:
    """ Extend a graph_pattern from a given endpoint variable by evaluating all
    possible candidate extensions on whether they satisfy the minimal support
    and confidence.

    :param pattern:
    :type pattern: GraphPattern
    :param a_i:
    :type a_i: Assertion
    :param a_j:
    :type a_j: Assertion
    :rtype: GraphPattern
    """

    # create new graph_pattern by extending that of the parent
    pattern_new = pattern.copy()
    pattern_new.parent = pattern
    pattern_new.extend(endpoint=endpoint, extension=extension)

    # compute support
    pattern_new.domain = pattern_new._update_domain()
    pattern_new.support = len(pattern_new.domain)

    return pattern_new

def init_root_patterns(rng:np.random.Generator, kg:KnowledgeGraph,
                       min_support:float, mode:Literal["A", "AT", "T"],
                       multimodal:bool) -> dict[str,list]:
    """ Creating all patterns of types which satisfy minimal support.

    :param rng:
    :type rng: np.random.Generator
    :param kg: 
    :type kg: KnowledgeGraph
    :param min_support:
    :type min_support: float
    :param mode:
    :type mode: Literal["A", "AT", "T"]
    :param multimodal:
    :type multimodal: bool
    :returns:
    :rtype: dict[str,list]
    """
    print("mapping root patterns")
    root_patterns = dict()

    # don't generate what we won't need
    generate_Abox = False
    generate_Tbox = False
    if "A" in mode:
        generate_Abox = True
    if "T" in mode:
        generate_Tbox = True

    # root node of graph pattern
    #root_var = Variable(IRIRef("graph://root"))

    rdf_type = RDF + "type"
    rdf_type_idx = kg.r2i[rdf_type]
    class_freq = kg.A[rdf_type_idx].sum(axis=0)
    class_idx_list = np.where(class_freq > 0)[0]
    for class_idx in class_idx_list:
        # if the number of type instances do not exceed the minimal support then
        # any pattern of this type will not either
        support = class_freq[class_idx]
        if support < min_support:
            continue

        class_name = kg.i2n[class_idx]
        root_patterns[class_name] = list()

        print(f" type {class_name}...", end='')

        # find all members of this class, retrieve all their predicate-objects
        class_members = kg.A[rdf_type_idx, :, class_idx]
        class_members_idx = np.where(class_members)[0]
        class_members_po = kg.A[:, class_members, :]

        root_var = ObjectTypeVariable(class_name)
        for p_idx in range(kg.num_relations):
            if p_idx == rdf_type_idx:
                continue
            
            p_freq = class_members_po[p_idx].sum()
            if p_freq < min_support:
                # if the number of entities of type t that have this predicate
                # is less than the minimal support, then the overall pattern
                # will have less as well
                continue
       
            p = kg.i2r[p_idx]

            # create graph_patterns for all predicate-object pairs
            # treat both entities and literals as node
            s_idx_list, o_idx_list = np.where(class_members_po[p_idx] == True)
            s_idx_list = class_members_idx[s_idx_list]
            o_type, is_literal = infer_type(kg, rdf_type_idx, o_idx_list[-1])
            if generate_Abox:
                if multimodal and is_literal:
                    o_freqs = Counter([kg.i2n[i].value for i in o_idx_list])
                    o_value_idx_map = {v:{i for i, o_idx in enumerate(o_idx_list)
                                          if kg.i2n[o_idx].value == v} for v in o_freqs.keys()}    
                else:  # is IRI
                    o_freqs = Counter(o_idx_list)
                    o_value_idx_map = {kg.i2n[k]:{i for i, o_idx in enumerate(o_idx_list)
                                          if k == o_idx} for k in o_freqs.keys()}
    
                for o_value, o_freq in o_freqs.items():
                    if o_freq < min_support:
                        continue
    
                    if not is_literal:
                        o_value = kg.i2n[o_value]

                    domain = {s_idx_list[i] for i in o_value_idx_map[o_value]}
                    inv_assertion_map = {o_idx_list[i]: domain for i in o_value_idx_map[o_value]}
                    
                    # create new graph_pattern
                    pattern = new_graph_pattern(root_var, p, o_value,
                                                o_type, domain, inv_assertion_map)
                    
                    if pattern is not None and pattern.support >= min_support:
                        root_patterns[class_name].append(pattern)
                            
    
            # add graph_patterns with variables as objects
            if generate_Tbox:
                # assume that all objects linked by the same relation are of
                # the same type. This may not always be true but it is usually
                # the case in well-engineered graphs. See this as an
                # optimization by approximation.
                pattern = None
                o_idx = o_idx_list[0]
                inv_assertion_map = {o_idx: {s_idx_list[i] for i, idx in enumerate(o_idx_list)
                                             if idx == o_idx} for o_idx in o_idx_list}
                if o_idx in kg.i2d.keys():
                    # object is literal
                    o_type = kg.i2d[o_idx]
                    var_o = DataTypeVariable(o_type)
    
                    pattern = new_var_graph_pattern(root_var, var_o,
                                                      set(s_idx_list), p,
                                                      inv_assertion_map)
                else:
                    # object is entity (or bnode or literal without type)
                    o_type_idx = np.where(kg.A[rdf_type_idx, o_idx, :])[0]
                    if len(o_type_idx) > 0:
                        o_type = kg.i2n[o_type_idx[0]]
                        var_o = ObjectTypeVariable(o_type)
    
                        pattern = new_var_graph_pattern(root_var, var_o,
                                                          set(s_idx_list), p,
                                                          inv_assertion_map)
    
                if pattern is not None and pattern.support >= min_support:
                    root_patterns[class_name].append(pattern)
    
            if multimodal:
                # assume that all objects linked by the same relation are of
                # the same type. This may not always be true but it is usually
                # the case in well-engineered graphs. See this as an
                # optimization by approximation.
                o_idx = o_idx_list[0]
                if o_idx in kg.i2d.keys():
                    # object is literal                    
                    o_type = kg.i2d[o_idx]
                    if o_type not in SUPPORTED_XSD_TYPES:
                        continue
    
                    o_values = [kg.i2n[i].value for i in o_idx_list]
                    if len(o_values) < min_support:
                        # if the full set does not exceed the threshold then nor
                        # will subsets thereof
                        continue

                    os_idx_map = dict(zip(o_idx_list, s_idx_list))
                    clusters = compute_clusters(rng, o_type,
                                                o_values,
                                                o_idx_list)
                    for members, cluster in clusters:
                        if o_type in set.union(XSD_NUMERIC,
                                               XSD_DATETIME,
                                               XSD_DATEFRAG):
                            var_o = MultiModalNumericVariable(o_type, cluster)
                        else:  # treat as strings
                            var_o = MultiModalStringVariable(o_type, cluster)
    
                        domain = {os_idx_map[i] for i in members}
                        inv_assertion_map = {o_idx:domain for o_idx in members}
                        pattern = new_mm_graph_pattern(root_var, var_o,
                                                       domain, p,
                                                       inv_assertion_map)
    
                        if pattern is not None\
                            and pattern.support >= min_support:
                            root_patterns[class_name].append(pattern)

        tree_size = len(root_patterns[class_name])
        print(" (+{} discovered)".format(tree_size))

        if tree_size <= 0:
            del root_patterns[class_name]

    return root_patterns

def infer_type(kg:KnowledgeGraph, rdf_type_idx:int,
               node_idx:int) -> tuple[Union[IRIRef, str], bool]:
    """ Infer the (data) type or language tag of a resource. Defaults to
        rdfs:Class if none can be inferred.

    :param kg:
    :type kg: KnowledgeGraph
    :param rdf_type_idx:
    :type rdf_type_idx: int
    :param node_idx:
    :type node_idx: int
    :rtype: IRIRef
    """
    if node_idx in kg.i2d.keys():
        # this is a literal node
        return kg.i2d[node_idx], True

    try:
        iri = kg.i2n[kg.A[rdf_type_idx, node_idx, :]][0]
    except:
        iri = RDFS + "Class"

    return iri, False

def new_graph_pattern(root_var, p:IRIRef, o_value:Union[IRIRef,rdfLiteral], o_type:IRIRef,
                      domain:set, inv_assertion_map:dict[int,set[int]]) -> GraphPattern:
    """ Create a new graph_pattern and compute members and metrics

    :param kg:
    :type kg: KnowledgeGraph
    :param parent:
    :type parent: Optional[GraphPattern]
    :param var:
    :type var: ObjectTypeVariable
    :param p_idx:
    :type p_idx: int
    :param o_idx:
    :type o_idx: int
    :param class_members_idx:
    :type class_members_idx: np.ndarray
    :rtype: GraphPattern
    :returns: an instance of type GraphPattern
    """
    assertion = Assertion(root_var, p, ResourceWrapper(o_value, o_type))
    assertion._inv_idx_map = inv_assertion_map
    
    graph_pattern = GraphPattern({assertion})
    graph_pattern.domain = domain
    graph_pattern.support = len(graph_pattern.domain)
  
    return graph_pattern 
    
def new_var_graph_pattern(root_var, 
                  var_o:Union[ObjectTypeVariable, DataTypeVariable], 
                  domain:set, p:IRIRef, 
                  inv_assertion_map:dict[int,set[int]]) -> GraphPattern:
    """ Create a new variable graph_pattern and compute members and metrics

    :param kg:
    :param parent:
    :type parent: Optional[GraphPattern]
    :param var:
    :type var: ObjectTypeVariable
    :param var_o:
    :type var_o: Union[ObjectTypeVariable, DataTypeVariable]
    :param class_members_idx:
    :type class_members_idx: np.ndarray
    :param s_idx_list:
    :type s_idx_list: np.ndarray
    :param p_idx:
    :type p_idx: int
    :rtype: GraphPattern
    :returns: an instance of type GraphPattern
    """
    # assume that all objects linked by the same relation are of the same type
    # this may not always be true but usually it is
    assertion = Assertion(root_var, p, var_o)
    assertion._inv_idx_map = inv_assertion_map
    
    graph_pattern = GraphPattern({assertion})
    graph_pattern.domain = domain
    graph_pattern.support = len(graph_pattern.domain)

    return graph_pattern


def new_mm_graph_pattern(root_var, 
                 var_o:Union[ObjectTypeVariable, DataTypeVariable], 
                         domain:set, p:IRIRef, inv_assertion_map:dict[int,set[int]])\
                         -> GraphPattern:
    """ Create a new multimodal graph_pattern and compute members and metrics

    :param kg:
    :param parent:
    :type parent: Optional[GraphPattern]
    :param var:
    :type var: ObjectTypeVariable
    :param var_o:
    :type var_o: Union[ObjectTypeVariable, DataTypeVariable]
    :param members:
    :type members: set
    :param class_members_idx:
    :type class_members_idx: np.ndarray
    :param p_idx:
    :type p_idx: int
    :rtype: GraphPattern
    :returns: an instance of type GraphPattern
    """
    # assume that all objects linked by the same relation are of the same type
    # this may not always be true but usually it is
    assertion = Assertion(root_var, p, var_o)
    assertion._inv_idx_map = inv_assertion_map

    graph_pattern = GraphPattern({assertion})
    graph_pattern.domain = domain
    graph_pattern.support = len(graph_pattern.domain)


    return graph_pattern 
