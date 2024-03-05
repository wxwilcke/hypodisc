#! /usr/bin/env python

from collections import Counter
from random import random
from multiprocessing import Manager
from typing import Literal, Optional, Union

import numpy as np
from hypodisc.data.utils import write_query

from rdf.formats import NTriples
from rdf.namespaces import RDF, RDFS
from rdf.terms import IRIRef
from rdf.terms import Literal as rdfLiteral

from hypodisc.core.utils import predict_hash
from hypodisc.data.graph import KnowledgeGraph
from hypodisc.core.structures import (Assertion, GraphPattern,
                                      ResourceWrapper,
                                      DataTypeVariable,
                                      Variable,
                                      MultiModalNumericVariable,
                                      MultiModalStringVariable,
                                      ObjectTypeVariable)
from hypodisc.multimodal.clustering import (compute_clusters,
                                            SUPPORTED_XSD_TYPES)
from hypodisc.multimodal.datatypes import (XSD_DATEFRAG, XSD_DATETIME,
                                           XSD_NUMERIC, XSD_STRING)

XSD_TEMPORAL = set.union(XSD_DATETIME, XSD_DATEFRAG)
IGNORE_PREDICATES = {RDF + 'type', RDFS + 'label'}


def generate(root_patterns: dict[str, list],
             depths: range, min_support: int,
             p_explore: float, p_extend: float,
             max_length: int, max_width: int,
             out_writer: Optional[NTriples],
             out_prefix_map: Optional[dict[str, str]],
             out_ns: Optional[IRIRef],
             strategy: Literal["BFS", "DFS"]) -> int:
    """ Generate all patterns up to and including a maximum depth which
        satisfy a minimal support.

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
    """

    if strategy == "BFS":
        return generate_bf(root_patterns=root_patterns,
                           depths=depths,
                           min_support=min_support,
                           p_explore=p_explore,
                           p_extend=p_extend,
                           max_length=max_length,
                           max_width=max_width,
                           out_writer=out_writer,
                           out_prefix_map=out_prefix_map,
                           out_ns=out_ns)
    else:  # DFS
        return generate_df(root_patterns=root_patterns,
                           depths=depths,
                           min_support=min_support,
                           p_explore=p_explore,
                           p_extend=p_extend,
                           max_length=max_length,
                           max_width=max_width,
                           out_writer=out_writer,
                           out_prefix_map=out_prefix_map,
                           out_ns=out_ns)


def generate_df(root_patterns: dict[str, list],
                depths: range, min_support: int,
                p_explore: float, p_extend: float,
                max_length: int, max_width: int,
                out_writer: Optional[NTriples],
                out_prefix_map: Optional[dict[str, str]],
                out_ns: Optional[IRIRef]) -> int:
    """ Generate all patterns up to and including a maximum depth which
        satisfy a minimal support, using a depth first approach. This
        approach uses less memory but does not have the anytime property.

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
    """

    patterns = set()
    num_patterns = 0
    for name in sorted(list(root_patterns.keys())):
        print(f"type {name}")

        visited = set()
        for depth in range(0, depths.stop):
            print(" exploring depth {} / {}".format(depth+1, depths.stop),
                  end=" ")

            if depth == 0:
                patterns = root_patterns[name]

            derivatives = set()
            for pattern in patterns:
                if len(pattern) >= max_length or pattern.width() >= max_width:
                    continue

                # Gather candidate endpoints at this depth.
                # Only consider unbound object type variables since we
                # cannot connect to literals or data type variables, and
                # bound entities already have a fixed context.
                if depth <= 0:
                    endpoints = {pattern.root}

                    # add these as parents for next depth
                    if isinstance(pattern.assertion.rhs, ObjectTypeVariable):
                        derivatives.add(pattern)
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

                        candidates.add((endpoint, extension))

                    extensions = explore(pattern,
                                         candidates=candidates,
                                         max_length=max_length,
                                         max_width=max_width,
                                         min_support=min_support)

                    if out_writer is not None and out_prefix_map is not None:
                        for pattern in extensions:
                            num_patterns = write_query(out_writer, pattern,
                                                       num_patterns, out_ns,
                                                       out_prefix_map)

                    derivatives |= extensions

            print("(+{} discovered)".format(len(derivatives)))

            # omit exhausted classes from next iteration
            if len(derivatives) > 0:
                patterns = {v for v in derivatives}
            else:
                break

    return num_patterns


def generate_bf(root_patterns: dict[str, list],
                depths: range, min_support: int,
                p_explore: float, p_extend: float,
                max_length: int, max_width: int,
                out_writer: Optional[NTriples],
                out_prefix_map: Optional[dict[str, str]],
                out_ns: Optional[IRIRef]) -> int:
    """ Generate all patterns up to and including a maximum depth which
        satisfy a minimal support, using a breadth first approach. This
        approach has the anytime property yet uses more memory.

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
    """

    parents = dict()
    num_patterns = 0
    for depth in range(0, depths.stop):
        print("exploring depth {} / {}".format(depth+1, depths.stop))

        if depth == 0:
            parents = root_patterns

        visited = set()
        derivatives = dict()
        for name in sorted(list(parents.keys())):
            print(f" type {name}", end=" ")

            patterns = parents[name]
            if depth > 0:
                patterns = parents.pop(name)

            derivatives[name] = set()
            for pattern in patterns:
                if len(pattern) >= max_length or pattern.width() >= max_width:
                    continue

                # Gather candidate endpoints at this depth.
                # Only consider unbound object type variables since we
                # cannot connect to literals or data type variables, and
                # bound entities already have a fixed context.
                if depth <= 0:
                    endpoints = {pattern.root}

                    # add these as parents for next depth
                    if isinstance(pattern.assertion.rhs, ObjectTypeVariable):
                        derivatives[name].add(pattern)
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

                        candidates.add((endpoint, extension))

                    extensions = explore(pattern,
                                         candidates=candidates,
                                         max_length=max_length,
                                         max_width=max_width,
                                         min_support=min_support)

                    if out_writer is not None and out_prefix_map is not None:
                        for pattern in extensions:
                            num_patterns = write_query(out_writer, pattern,
                                                       num_patterns, out_ns,
                                                       out_prefix_map)

                    derivatives[name] |= extensions

            print("(+{} discovered)".format(len(derivatives[name])))

        # omit exhausted classes from next iteration
        parents = {k: v for k, v in derivatives.items()
                   if len(v) > 0}

    return num_patterns


def explore(parent: GraphPattern, candidates: set,
            max_length: int, max_width: int, min_support: int) -> set:
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

                # add as new if satisfies support and has a more constraint
                # domain than its parent or when the extension allows for
                # possible future extensions.
                if pattern_new.support >= min_support\
                        and (isinstance(extension.rhs, ObjectTypeVariable)
                             or pattern_new.parent is None
                             or pattern_new.support <
                             pattern_new.parent.support):
                    qexplore.put(pattern_new)  # explore further extensions
                    derivatives.add(pattern_new)
                else:
                    skipped.add(pattern_hash)

    return derivatives


def extend(pattern: GraphPattern, endpoint: Variable, extension: Assertion)\
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


def init_root_patterns(rng: np.random.Generator, kg: KnowledgeGraph,
                       min_support: float, mode: Literal["A", "AT", "T"],
                       textual_support: bool, numerical_support: bool,
                       temporal_support: bool, exclude: list[str])\
            -> dict[str, list]:
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

    multimodal = textual_support | numerical_support | temporal_support

    rdf_type = RDF + "type"
    rdf_type_idx = kg.r2i[rdf_type]

    A_type = kg.A[rdf_type_idx]

    # list of predicate indices to ingore
    exclude_idx = [kg.r2i[IRIRef(p)] for p in exclude]

    class_idx_list = sorted(list(set(A_type.col)))
    class_freq = A_type.sum(axis=0)[class_idx_list]
    for class_i, class_idx in enumerate(class_idx_list):
        # if the number of type instances do not exceed the minimal support
        # then any pattern of this type will not either
        support = class_freq[class_i]
        if support < min_support:
            continue

        class_name = kg.i2n[class_idx]
        root_patterns[class_name] = list()

        print(f" type {class_name}...", end='')

        # find all members of this class,
        class_members_idx = A_type.row[A_type.col == class_idx]

        root_var = ObjectTypeVariable(class_name)
        for p_idx in range(kg.num_relations):
            if p_idx == rdf_type_idx or p_idx in exclude_idx:
                continue

            # mask for class members that have this relation outgoing
            p_mask = np.isin(kg.A[p_idx].row, class_members_idx)
            if sum(p_mask) < min_support:
                # if the number of entities of type t that have this predicate
                # is less than the minimal support, then the overall pattern
                # will have less as well
                continue

            p = kg.i2r[p_idx]

            # list of global indices for class members with this property
            # plus the matching tail nodes
            s_idx_list = kg.A[p_idx].row[p_mask]
            o_idx_list = kg.A[p_idx].col[p_mask]

            # infer (data) type from single tail node (assume rest is same)
            o_type, is_literal = infer_type(kg, rdf_type_idx, o_idx_list[-1])

            # assume that all objects linked by the same relation are of
            # the same type. This may not always be true but it is usually
            # the case in well-engineered graphs. See this as an
            # optimization by approximation.
            pattern = None
            o_idx = o_idx_list[0]
            inv_assertion_map = {o_idx: {s_idx_list[i]
                                         for i, idx in
                                         enumerate(o_idx_list)
                                         if idx == o_idx}
                                 for o_idx in o_idx_list}
            if generate_Tbox and o_idx in kg.ni2ai.keys():
                # object is literal
                o_type = kg.i2a[kg.ni2ai[o_idx]]
                var_o = DataTypeVariable(o_type)

                pattern = new_var_graph_pattern(root_var, var_o,
                                                set(s_idx_list), p,
                                                inv_assertion_map)
            else:
                # object is entity (or bnode or literal without type)
                idx = np.where(A_type.row == o_idx)
                o_type_idx = A_type.col[idx]
                if len(o_type_idx) > 0:
                    o_type = kg.i2n[o_type_idx[0]]
                    var_o = ObjectTypeVariable(o_type)

                    pattern = new_var_graph_pattern(root_var, var_o,
                                                    set(s_idx_list), p,
                                                    inv_assertion_map)

            if pattern is not None and pattern.support >= min_support:
                root_patterns[class_name].append(pattern)

            # create graph_patterns for all predicate-object pairs
            # treat both entities and literals as node
            if generate_Abox:
                if multimodal and is_literal:
                    o_freqs = Counter([kg.i2n[i].value for i in o_idx_list])
                    o_value_idx_map = {v:
                                       {i for i, o_idx in enumerate(o_idx_list)
                                        if kg.i2n[o_idx].value == v}
                                       for v in o_freqs.keys()}
                else:  # if IRI
                    o_freqs = Counter(o_idx_list)
                    o_value_idx_map = {kg.i2n[k]:
                                       {i for i, o_idx in enumerate(o_idx_list)
                                        if k == o_idx} for k in o_freqs.keys()}

                for o_value, o_freq in o_freqs.items():
                    if o_freq < min_support:
                        continue

                    if not is_literal:
                        o_value = kg.i2n[o_value]

                    domain = {s_idx_list[i] for i in o_value_idx_map[o_value]}
                    inv_assertion_map = {o_idx_list[i]: domain
                                         for i in o_value_idx_map[o_value]}

                    # create new graph_pattern
                    pattern = new_graph_pattern(root_var, p, o_value,
                                                o_type, domain,
                                                inv_assertion_map)

                    if pattern is not None and pattern.support >= min_support:
                        root_patterns[class_name].append(pattern)

            if multimodal:
                # assume that all objects linked by the same relation are of
                # the same type. This may not always be true but it is usually
                # the case in well-engineered graphs. See this as an
                # optimization by approximation.
                o_idx = o_idx_list[-1]
                if o_idx in kg.ni2ai.keys():
                    # object is literal
                    o_type = kg.i2a[kg.ni2ai[o_idx]]
                    if o_type not in SUPPORTED_XSD_TYPES:
                        continue

                    if not textual_support and o_type in XSD_STRING:
                        continue
                    if not numerical_support and o_type in XSD_NUMERIC:
                        continue
                    if not temporal_support and o_type in XSD_TEMPORAL:
                        continue

                    o_values = [kg.i2n[i].value for i in o_idx_list]
                    if len(o_values) < min_support:
                        # if the full set does not exceed the threshold then
                        # nor will subsets thereof
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
                        inv_assertion_map = {o_idx: domain
                                             for o_idx in members}
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


def infer_type(kg: KnowledgeGraph, rdf_type_idx: int,
               node_idx: int) -> tuple[Union[IRIRef, str], bool]:
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
    if node_idx in kg.ni2ai.keys():
        # this is a literal node
        return kg.i2a[kg.ni2ai[node_idx]], True

    try:
        idx = np.where(kg.A[rdf_type_idx].row == node_idx)
        class_idx = kg.A[rdf_type_idx].col[idx][0]
        iri = kg.i2n[class_idx]
    except IndexError:
        iri = RDFS + "Class"

    return iri, False


def new_graph_pattern(root_var, p: IRIRef, o_value: Union[IRIRef, rdfLiteral],
                      o_type: IRIRef, domain: set,
                      inv_assertion_map: dict[int, set[int]]) -> GraphPattern:
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
                          var_o: Union[ObjectTypeVariable, DataTypeVariable],
                          domain: set, p: IRIRef,
                          inv_assertion_map: dict[int, set[int]])\
                                  -> GraphPattern:
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
                         var_o: Union[ObjectTypeVariable, DataTypeVariable],
                         domain: set, p: IRIRef,
                         inv_assertion_map: dict[int, set[int]])\
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
