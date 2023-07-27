#! /usr/bin/env python

from random import random, choice
from time import time
from multiprocessing import Manager
from typing import Literal, Optional, Union

import numpy as np
from hypodisc.data.graph import KnowledgeGraph

from rdf.namespaces import RDF, RDFS, XSD
from rdf.terms import IRIRef

from hypodisc.structures import (Assertion, Clause, ClauseBody, ResourceWrapper, TypeVariable,
                                 DataTypeVariable, IdentityAssertion,
                                 MultiModalVariable,
                                 MultiModalDateFragVariable, MultiModalDateTimeVariable,
                                 MultiModalNumericVariable, MultiModalStringVariable,
                                 ObjectTypeVariable, GenerationForest, GenerationTree)
from hypodisc.multimodal import (compute_clusters, SUPPORTED_XSD_TYPES, XSD_DATEFRAG,
                                 XSD_DATETIME, XSD_NUMERIC, XSD_STRING)
#from hypodisc.utils import cast_xsd


IGNORE_PREDICATES = {RDF + 'type', RDFS + 'label'}
IDENTITY = IRIRef("local://identity")  # reflexive property

def generate(rng:np.random.Generator, kg:KnowledgeGraph,
             depths:range, min_support:int, min_confidence:int,
             p_explore:float, p_extend:float,
             max_length:int, max_width:int,
             prune:bool, multimodal:bool,
             mode:Literal["AA", "AT", "TA", "TT",
             "AB", "BA", "TB", "BT", "BB"]) -> GenerationForest:
    """ Generate all clauses up to and including a maximum depth which satisfy a minimal
    support and confidence.

    :param kg:
    :type kg: KnowledgeGraph
    :param depths:
    :type depths: range
    :param min_support:
    :type min_support: int
    :param min_confidence:
    :type min_confidence: int
    :param p_explore:
    :type p_explore: float
    :param p_extend:
    :type p_extend: float
    :param max_length:
    :type max_length: int
    :param max_width:
    :type max_width: int
    :param prune:
    :type prune: bool
    :param multimodal:
    :type multimodal: bool
    :param mode:
    :type mode: Literal["AA", "AT", "TA", "TT",
                 "AB", "BA", "TB", "BT", "BB"]
    :rtype: GenerationForest
    """

    t0 = time()
    generation_forest = init_generation_forest(rng, kg, min_support,
                                               min_confidence,
                                               mode, multimodal)

    mode_skip_dict = dict()
    npruned = 0
    for depth in range(0, depths.stop):
        print("generating depth {} / {}".format(depth + 1, depths.stop))

        for class_name in generation_forest.types:
            print(f" type {class_name}", end=" ")

            derivatives = set()  # clauses derived from parents of depth d
            prune_set = set()

            for phi in generation_forest.get_tree(class_name).get(depth):
                #if depth == 0 and prune and\
                #   (isinstance(clause.head.rhs, ObjectTypeVariable) or
                #    isinstance(clause.head.rhs, DataTypeVariable)):
                #    # assume that predicate range is consistent irrespective of
                #    # context beyond depth 0
                #    npruned += 1

                #    continue

                #if depth == 0 and mode[0] != mode[1] and \
                #   (mode[0] == "A" and isinstance(phi.head.rhs, TypeVariable) or
                #    mode[0] == "T" and not isinstance(phi.head.rhs, TypeVariable)):
                #    # skip clauses with Abox or Tbox heads to filter
                #    # exploration on the remainder from depth 0 and 'up'
                #    if class_name not in mode_skip_dict.keys():
                #        mode_skip_dict[class_name] = set()
                #    mode_skip_dict[class_name].add(phi)

                #    continue

                if len(phi) < max_length and phi.width() < max_width:
                    candidates = list()

                    # Gather candidate endpoints at this depth.
                    # Only consider unbound object type variables since we
                    # cannot connect to literals or data type variables, and
                    # bound entities already have a fixed context.
                    endpoints = {a for a in phi.body.distances[depth]
                                 if isinstance(a.rhs, ObjectTypeVariable)}

                    for a_i in endpoints:
                        if a_i.rhs.value not in generation_forest.types:
                            # if the type lacks support, then any clause
                            # which incorporates it will too
                            continue

                        if p_explore < random():
                            # skip this endpoint with probability p_explore
                            continue

                        # Gather all candidate extensions that can connect
                        # to an object type variable of the relevant type.
                        # These are the level 0 heads found earlier.
                        tree = generation_forest.get_tree(a_i.rhs.value)
                        for psi in tree.get(0):
                            if p_extend < random():
                                # skip this extension with probability p_extend
                                continue

                            a_j = psi.head
                            if mode[1] == "A"\
                                    and isinstance(a_j.rhs, TypeVariable):
                                # limit body extensions to Abox
                                continue
                            if mode[1] == "T"\
                                    and not isinstance(a_j.rhs, TypeVariable):
                                # limit body extensions to Tbox
                                continue
                            if isinstance(a_j.rhs, MultiModalVariable):
                                # don't allow multimodal variables in body
                                continue

                            candidates.append(((a_i, a_j), psi.satisfy_complete))

                    derivatives |= explore(kg = kg,
                                           phi = phi,
                                           candidates = candidates,
                                           depth = depth,
                                           max_length = max_length,
                                           max_width = max_width,
                                           min_support = min_support,
                                           min_confidence = min_confidence,
                                           mode = mode,
                                           prune = prune)

                # clear domain of clause (which we won't need anymore) to save memory
                #phi.satisfy_body = None
                #phi.satisfy_complete = None

           #     if prune and depth > 0 and phi._prune is True:
           #         prune_set.add(phi)

           # # prune clauses after generating children to still allow for complex children
           # if prune:
           #     generation_forest.prune(class_name, depth, prune_set)
           #     npruned += len(prune_set)

           #     # prune children in last iteration
           #     if depth == depths.stop-1:
           #         prune_set = set()
           #         for derivative in derivatives:
           #             if derivative._prune is True:
           #                 prune_set.add(derivative)

           #         derivatives -= prune_set
           #         npruned += len(prune_set)

            print("(+{} added)".format(len(derivatives)))

            # remove clauses after generating children if we are
            # not interested in previous depth
            #if depth > 0 and depth not in depths:
            #    n0 = generation_forest.get_tree(class_name).size
            #    generation_forest.clear(class_name, depth)

            #    npruned += n0 - generation_forest.get_tree(class_name).size

            generation_forest.update_tree(class_name, derivatives, depth+1)

    #if len(mode_skip_dict) > 0:
    #    # prune unwanted clauses at depth 0 now that we don't need them anymore
    #    for class_name, skip_set in mode_skip_dict.items():
    #        generation_forest.prune(class_name, 0, skip_set)

    #if 0 not in depths and depths.stop-depths.start > 0:
    #    for class_name in generation_forest.types:
    #        n0 = generation_forest.get_tree(class_name).size
    #        generation_forest.clear(class_name, 0)

    #        npruned += n0 - generation_forest.get_tree(class_name).size

    duration = time()-t0
    print('generated {} clauses in {:0.3f}s'.format(
        sum([tree.size for tree in generation_forest._trees.values()]),
        duration),
    end="")

    #if npruned > 0:
    #    print(" ({} pruned)".format(npruned))
    #else:
    #    print()

    return generation_forest

#def visited(V, body, a_i, a_j):
#    body.extend(endpoint=a_i, extension=a_j)
#    return body in V
#
#def covers(body, a_i, a_j):
#    return a_j in body.connections[hash(a_i)]
#
#def bad_combo(U, body, a_i, a_j):
#    body.extend(endpoint=a_i, extension=a_j)
#    return body in U

def explore(kg:KnowledgeGraph, phi:Clause, candidates:list,
            depth:int, max_length:int, max_width:int,
            min_support:int, min_confidence:int,
            mode:Literal["AA", "AT", "TA", "TT",
                         "AB", "BA", "TB", "BT", "BB"],
            prune:bool) -> set:
    """ Explore all predicate-object pairs which where added by the previous
    iteration as possible endpoints to expand from.
    """
    derivatives = set()  # clauses derived from parent
    seen = set()  # visited clauses
    blacklist = set()  # bad combinations

    with Manager() as manager:
        qexplore = manager.Queue()
        qexplore.put(phi)
        while not qexplore.empty():
            psi = qexplore.get()

            if len(psi) >= max_length or psi.width() >= max_width:
                continue

            #if depth+1 in psi.body.distances.keys():
            #    if len(psi.body.distances[depth+1]) >= max_width:
            #        continue

            # skip with probability of (1 - p_explore)
            #skip_endpoint = None
            #if p_explore < random():
            #    skip_endpoint = choice(tuple(candidates))

            for (a_i, a_j), a_j_domain in candidates:
                # test identity here as endpoint is same object
                #if a_j is skip_endpoint:
                #    continue

                # skip with probability of (1 - p_extend)
                # place it here as we only want to skip those we are really adding
                #if p_extend < random():
                #    continue

                #if visited(seen, psi.body.copy(), a_i, a_j)\
                #   or covers(psi.body, a_i, a_j)\
                #   or bad_combo(blacklist, psi.body.copy(), a_i, a_j):
                #    continue

                chi = extend(psi, a_i, a_j, a_j_domain, depth,
                             min_support, min_confidence)

                if chi is not None:
                    qexplore.put(chi)
                    derivatives.add(chi)
                    #seen.add(chi.body.copy())

                #else:
                #    blacklist.add((psi.body.copy(), a_i, a_j))

        if len(derivatives) <= 0 or not prune:
            return derivatives

        # set delayed pruning on siblings if all have same support/confidence
        # (ie, as it doesn't matter which extension we add, we can assume that none really matter)
#        qprune = manager.Queue()
#        qprune.put(phi)
#        while not qprune.empty():
#            psi = qprune.get()
#            scores_set = list()
#            for chi in psi.children:
#                scores_set.append((chi.support, chi.confidence))
#
#                if len(chi.children) > 0:
#                    qprune.put(chi)
#
#            if len(psi.children) >= 2\
#               and scores_set.count(scores_set)[0] == len(scores_set):
#                for chi in psi.children:
#                    chi._prune = True
#
    return derivatives

def extend(psi:Clause, a_i:Assertion, a_j:Assertion, 
           a_j_domain:set, depth:int, min_support:int,
           min_confidence:int) -> Union[Clause, None]:
    """ Extend a clause from a given endpoint variable by evaluating all
    possible candidate extensions on whether they satisfy the minimal support
    and confidence.
    """

    # omit if candidate for level 0 is equivalent to head
    if depth <= 0 and psi.head.equiv(a_j):
        return None

    # omit equivalents on same context level (exact or by type)
    if depth+1 in psi.body.distances.keys():
        for assertion in psi.body.distances[depth+1]:
            if assertion.equiv(a_j):
                return None

    # create new clause body by extending that of the parent
    head = psi.head
    body = psi.body.copy()
    body.extend(endpoint=a_i, extension=a_j)

    # compute support
    # intersection between support of parent and that of extension
    satisfies_body = set.intersection(psi.satisfy_body, a_j_domain)
    support = len(satisfies_body)
    if support < min_support:
        return None

    # compute confidence
    # intersection between the confidence of parent and the now reduced domain
    # that is represented by the body.
    satisfies_complete = set.intersection(psi.satisfy_complete,
                                            satisfies_body)
    confidence = len(satisfies_complete)
    if confidence < min_confidence:
        return None

    # compute probabilities
    probability = confidence / support

    # save more constraint clause
    chi = Clause(head = head,
                 body = body,
                 parent = psi,
                 satisfy_body = satisfies_body,
                 satisfy_complete = satisfies_complete,
                 support = support,
                 confidence = confidence,
                 probability = probability)

    # set delayed pruning if no reduction in domain
    #if support >= psi.support:
    #    chi._prune = True

    return chi

def init_generation_forest(rng:np.random.Generator, kg:KnowledgeGraph,
                           min_support:float, min_confidence:float,
                           mode:Literal["AA", "AT", "TA", "TT",
                                        "AB", "BA", "TB", "BT", "BB"],
                           multimodal:bool) -> GenerationForest:
    """ Initialize the generation forest by creating all generation trees of
    types which satisfy minimal support and confidence.

    :param rng:
    :type rng: np.random.Generator
    :param kg: 
    :type kg: KnowledgeGraph
    :param min_support:
    :type min_support: float
    :param min_confidence:
    :type min_confidence: float
    :param mode:
    :type mode: Literal["AA", "AT", "TA", "TT",
                        "AB", "BA", "TB", "BT", "BB"]
    :param multimodal:
    :type multimodal: bool
    :returns:
    :rtype: GenerationForest
    """
    print("initializing Generation Forest")
    generation_forest = GenerationForest()

    # don't generate what we won't need
    generate_Abox_heads = True
    generate_Tbox_heads = True
    if mode == "AA":
        generate_Tbox_heads = False
    elif mode == "TT":
        generate_Abox_heads = False

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

        class_node = kg.i2n[class_idx]
        print(f" initializing Generation Tree for type {class_node} ", end=" ")
        # find all members of this class, retrieve all their predicate-objects
        class_members = kg.A[rdf_type_idx, :, class_idx]
        class_members_idx = get_boolean_array_true_index(class_members)
        class_members_po = kg.A[:, class_members, :]

        # create shared variables
        parent = None 
        var = ObjectTypeVariable(resource=class_node)

        # generate clauses for each predicate-object pair
        generation_tree = GenerationTree(class_node)
        for p_idx in range(kg.num_relations):
            if p_idx == rdf_type_idx:
                # skip rdf type relation
                continue

            p_freq = class_members_po[p_idx].sum()
            if p_freq < min_support:
                # if the number of entities of type t that have this predicate
                # is less than the minimal support, then the overall pattern
                # will have less as well
                continue

            # create clauses for all predicate-object pairs
            # treat both entities and literals as node
            s_idx_list, o_idx_list = np.where(class_members_po[p_idx] == True)
            o_type = infer_type(kg, rdf_type_idx, o_idx_list[-1])
            if generate_Abox_heads:
                seen = set()
                for o_idx in o_idx_list:
                    if o_idx in seen:
                        # avoid duplicates
                        continue

                    # create new clause
                    phi = new_clause(kg, parent, var, p_idx, o_idx, o_type,
                                     class_members_idx)

                    if phi is not None and phi.confidence >= min_confidence:
                        generation_tree.add(phi, depth=0)

                    seen.add(o_idx)

            # add clauses with variables as objects
            if generate_Tbox_heads:
                # assume that all objects linked by the same relation are of
                # the same type. This may not always be true but it is usually
                # the case in well-engineered graphs. See this as an
                # optimization by approximation.
                phi = None
                o_idx = o_idx_list[0]
                if o_idx in kg.i2d.keys():
                    # object is literal
                    o_type = kg.i2d[o_idx]
                    var_o = DataTypeVariable(o_type)

                    phi = new_varclause(kg, parent, var, var_o,
                                        class_members_idx, s_idx_list,
                                        p_idx)
                else:
                    # object is entity (or bnode or literal without type)
                    o_type_idx =\
                        get_boolean_array_true_index(kg.A[rdf_type_idx,
                                                          o_idx, :])
                    if len(o_type_idx) > 0:
                        o_type = kg.i2n[o_type_idx[0]]
                        var_o = ObjectTypeVariable(o_type)

                        phi = new_varclause(kg, parent, var, var_o,
                                            class_members_idx, s_idx_list,
                                            p_idx)

                if phi is not None and phi.confidence >= min_confidence:
                    generation_tree.add(phi, depth=0)


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

                    o_values = [kg.i2n[i] for i in o_idx_list]
                    if len(o_values) < min_confidence:
                        # if the full set does not exceed the threshold then nor
                        # will subsets thereof
                        continue

                    clusters = compute_clusters(rng, o_type, o_values)
                    for cluster in clusters:
                        if o_type in XSD_NUMERIC:
                            var_o = MultiModalNumericVariable(o_type, cluster)
                        elif o_type in XSD_DATETIME:
                            var_o = MultiModalDateTimeVariable(o_type, cluster)
                        elif o_type in XSD_DATEFRAG:
                            var_o = MultiModalDateFragVariable(o_type, cluster)
                        else:  # treat as strings
                            var_o = MultiModalStringVariable(o_type, cluster)

                        phi = new_mmclause(kg, parent, var, var_o,
                                            class_members_idx, s_idx_list,
                                            p_idx)

                        if phi is not None and phi.confidence >= min_confidence:
                            generation_tree.add(phi, depth=0)

        print("done (+{} added)".format(generation_tree.size))

        if generation_tree.size <= 0:
            continue

        generation_forest.plant(class_node, generation_tree)

    return generation_forest

def infer_type(kg:KnowledgeGraph, rdf_type_idx:int,
               node_idx:int) -> IRIRef:
    """ Infer the (data) type of a resource. Defaults to rdfs:Class if none can
        be inferred.

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
        return kg.i2d[node_idx]

    try:
        iri = kg.i2n[kg.A[rdf_type_idx, node_idx, :]][0]
    except:
        iri = RDFS + "Class"

    return iri

def new_clause(kg:KnowledgeGraph, parent:Optional[Clause],
               var:ObjectTypeVariable, p_idx:int,
               o_idx:int, o_type:IRIRef,
               class_members_idx:np.ndarray) -> Clause:
    """ Create a new clause and compute members and metrics

    :param kg:
    :type kg: KnowledgeGraph
    :param parent:
    :type parent: Optional[Clause]
    :param var:
    :type var: ObjectTypeVariable
    :param p_idx:
    :type p_idx: int
    :param o_idx:
    :type o_idx: int
    :param class_members_idx:
    :type class_members_idx: np.ndarray
    :rtype: Clause
    :returns: an instance of type Clause
    """
    # indices of members in the domain who satisfy body and/or head
    satisfy_complete = set(class_members_idx[kg.A[p_idx,
                                                  class_members_idx,
                                                  o_idx]])
    satisfy_body = set(class_members_idx)

    # number of members in the domain who satisfy body and/or head
    support = len(satisfy_body)
    confidence = len(satisfy_complete)

    # probability that an arbitrary member of the domain satisfies the head
    probability = confidence/support

    return Clause(head=Assertion(var, kg.i2r[p_idx],
                                 ResourceWrapper(kg.i2n[o_idx], type=o_type)),
                  body=ClauseBody(IdentityAssertion(var, IDENTITY, var)),
                  support=support,
                  confidence=confidence,
                  probability=probability,
                  satisfy_body=satisfy_body,
                  satisfy_complete=satisfy_complete,
                  parent=parent)

def new_varclause(kg, parent:Optional[Clause], var:ObjectTypeVariable,
                  var_o:Union[ObjectTypeVariable, DataTypeVariable], 
                  class_members_idx:np.ndarray,
                  s_idx_list:np.ndarray, p_idx:int) -> Clause:
    """ Create a new variable clause and compute members and metrics

    :param kg:
    :param parent:
    :type parent: Optional[Clause]
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
    :rtype: Clause
    :returns: an instance of type Clause
    """
    # indices of members in the domain who satisfy body and/or head
    # assume that all objects linked by the same relation are of the same type
    # this may not always be true but usually it is
    satisfy_complete = set(class_members_idx[s_idx_list])
    satisfy_body = set(class_members_idx)

    # number of members in the domain who satisfy body and/or head
    support = len(satisfy_body)
    confidence = len(satisfy_complete)

    # probability that an arbitrary member of the domain satisfies head
    probability = confidence/support

    return Clause(head=Assertion(var, kg.i2r[p_idx], var_o),
                  body=ClauseBody(IdentityAssertion(var, IDENTITY, var)),
                  support=support,
                  confidence=confidence,
                  probability=probability,
                  satisfy_body=satisfy_body,
                  satisfy_complete=satisfy_complete,
                  parent=parent)

def new_mmclause(kg, parent:Optional[Clause], var:ObjectTypeVariable,
                 var_o:Union[ObjectTypeVariable, DataTypeVariable], 
                 class_members_idx:np.ndarray,
                 s_idx_list:np.ndarray, p_idx:int) -> Clause:
    """ Create a new multimodal clause and compute members and metrics

    :param kg:
    :param parent:
    :type parent: Optional[Clause]
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
    :rtype: Clause
    :returns: an instance of type Clause
    """
    # indices of members in the domain who satisfy body and/or head
    # assume that all objects linked by the same relation are of the same type
    # this may not always be true but usually it is
    satisfy_complete = set(class_members_idx[s_idx_list])
    satisfy_body = set(class_members_idx)

    # number of members in the domain who satisfy body and/or head
    support = len(satisfy_body)
    confidence = len(satisfy_complete)

    # probability that an arbitrary member of the domain satisfies the head
    probability = confidence/support

    return Clause(head=Assertion(var, kg.i2r[p_idx], var_o),
                  body=ClauseBody(IdentityAssertion(var, IDENTITY, var)),
                  support=support,
                  confidence=confidence,
                  probability=probability,
                  satisfy_body=satisfy_body,
                  satisfy_complete=satisfy_complete,
                  parent=parent)

def get_boolean_array_true_index(bool_array: np.ndarray) -> np.ndarray:
    """ return index where boolean array is true

    :param bool_array:
    :type bool_array: np.ndarray
    :rtype: np.ndarray
    :returns:
    """
    return np.arange(len(bool_array))[bool_array]
