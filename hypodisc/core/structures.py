#! /usr/bin/env python

from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Set, Union
from uuid import uuid4

from rdf.terms import IRIRef, Resource


class GraphPattern():
    """ GraphPattern class

    Holds all assertions of a graph pattern and keeps
    track of the connections and distances (from the root) of these assertions.
    """

    def __init__(self,
                 assertions:set[Assertion] = set(),
                 domain:set = set(),
                 parent:Optional[GraphPattern] = None) -> None:
        """__init__.

        :param identity:
        :type identity: IdentityAssertion
        :rtype: None
        """
        self.domain = domain  # type:set
        self.support = len(self.domain)
        self.parent = parent

        self.connections = dict()  # type: Dict[Assertion, set]
        self.distances = dict()  # type: Dict[int, set]
        self.distances_reverse = dict()  # type: Dict[Assertion, int]

        if len(assertions) > 0:
            self.root = self._infer_root(assertions)
            self.distances = self._compute_distances({self.root}, assertions)
            self.distances_reverse = {a:d for d,a_set in self.distances.items()\
                    for a in a_set}
            self.connections = self._compute_connections(self.distances)

            if len(assertions) == 1:
                self.assertion = list(self.distances[0])[0]

    def _infer_root(self, assertions:set[Assertion]) -> Variable:
        """ Infer the root node of the graph

        :param assertions:
        :type assertions: set[Assertion]
        :rtype: Variable
        """
        # infer root variable
        vars_head = set()
        vars_tail = set()
        for lhs, _, rhs in assertions:
            if isinstance(lhs, Variable):
                vars_head.add(lhs)
            if isinstance(rhs, Variable):
                vars_tail.add(rhs)

        return (vars_head - vars_tail).pop()

    def _compute_distances(self, targets:set[Variable],
                           assertions:set[Assertion], depth:int = 0)\
                                   -> dict[int, set[Assertion]]:
        """ Compute the number of hops from the root node

        :param targets:
        :type targets: set[Variable]
        :param assertions:
        :type assertions: set[Assertion]
        :param depth:
        :type depth: int
        :rtype: dict[int, set[Assertion]]
        """
        if len(assertions) <= 0:
            return dict()

        distances = {depth: set()}
        remainder = set()
        endpoints = set()
        for a in assertions:
            if a.lhs in targets:
                distances[depth].add(a)

                if isinstance(a.rhs, Variable):
                    endpoints.add(a.rhs)

                continue

            remainder.add(a)

        return distances | self._compute_distances(endpoints,
                                                   remainder,
                                                   depth + 1)

    def _compute_connections(self, distances:dict[int, set[Assertion]])\
            -> dict[Assertion, set[Assertion]]:
        """ Compute the connections between statements

        :param distances:
        :type distances: dict[int, set[Assertion]]
        :rtype: dict[Assertion, set[Assertion]]
        """
        connections = {a: set() for a_set in distances.values() for a in a_set}
        varmap = {a.rhs: a for a in connections.keys()}
        for d, a_set in distances.items():
            if d <= 0:
                continue

            for a in a_set:
                a_key = varmap[a.lhs]
                connections[a_key].add(a)

        return connections

    def _update_domain(self, assertion:Optional[Assertion] = None) -> set:
        domains = list()  # type: list[set[int]]
        if assertion is None:
            for rooted_assertion in self.distances[0]:
                domains.append(self._update_domain(rooted_assertion))

            return set.intersection(*domains)

        if len(self.connections[assertion]) <= 0:
            # if there are no further connections then we only
            # care about the domain
            for dict_values in assertion._inv_idx_map.values():
                domains.append(set(dict_values))

            return set.union(*domains)

        arange = list()
        for connection in self.connections[assertion]:
             arange.append(self._update_domain(connection))
        arange = set.intersection(*arange)

        for obj, subs in assertion._inv_idx_map.items():
            if obj in arange:
                domains.append(subs)

        return set.union(*domains) if len(domains) > 0 else set()


    def extend(self, endpoint:Variable, extension:Assertion) -> None:
        """ Extend the body by appending a new assertion to an existing
        endpoint.

        :param endpoint:
        :type endpoint: Assertion
        :param extension:
        :type extension: Assertion
        :rtype: None
        """
        if type(endpoint) is not ObjectTypeVariable:
            raise Exception('Cannot extend from this endpoint')
        if endpoint.value != extension.lhs.value:
            raise Exception('Cannot connect different classes')

        if endpoint == self.root:
            assert extension.lhs == endpoint
            distance = 0
        else:
            assertion = None
            for a in self.connections.keys():
                if a.rhs == endpoint:
                    assertion = a

                    break

            if assertion is None:
                raise Exception('No possible connection found')

            # connect nodes
            inv_idx_map = extension._inv_idx_map
            extension = Assertion(assertion.rhs,
                                  extension.predicate,
                                  extension.rhs)
            extension._inv_idx_map = inv_idx_map

            self.connections[assertion].add(extension)
            distance = self.distances_reverse[assertion] + 1

        self.connections[extension] = set()
        self.distances_reverse[extension] = distance
        if distance not in self.distances.keys():
            self.distances[distance] = set()
        self.distances[distance].add(extension)

    def copy(self) -> GraphPattern:
        """ Create a deep copy, except for the assertions, which remain
        as pointers.

        :rtype: GraphPattern
        """
        g = GraphPattern(domain = {v for v in self.domain})

        g.root = self.root
        g.assertion = self.assertion
        g.domain = {e for e in self.domain}
        g.support = self.support

        g.connections = {k: {v for v in self.connections[k]}
                         for k in self.connections.keys()}
        g.distances = {k: {v for v in self.distances[k]}
                       for k in self.distances.keys()}
        g.distances_reverse = {k: v for k,v in self.distances_reverse.items()}

        return g

    def __len__(self) -> int:
        """ Return the number of assertions

        :rtype: int
        """
        return len(self.connections.keys())

    def width(self, depth:Optional[int]=None) -> int:
        """Return the maximum number of assertions on a certain depth, or the
        most overall if depth is None.

        :rtype: int
        """
        if depth is None:
            return max([len(s) for s in self.distances.values()])
        else:
            return len(self.distances[depth])

    def __contains__(self, assertion:Assertion) -> bool:
        for a in self.connections.keys():
            if a.equiv(assertion):
                return True

        return False

    def contains_at_depth(self, assertion:Assertion, depth:int) -> bool:
        if depth in self.distances.keys():
            for a in self.distances[depth]:
                if a.equiv(assertion):
                    return True

        return False

    def depth(self) -> int:
        """ Return the length of the longest non-cyclic path

        :rtype: int
        """
        return max(self.distances.keys())

    def __lt__(self, other) -> bool:
        """ Return true if self has less assertions or less depth

        :param other:
        :rtype: bool
        """
        if len(self) < len(other):
            return True
        if max(self.distances.keys()) < max(other.distances.keys()):
            return True

        return False

    def __repr__(self) -> str:
        """ Return an internal string representation

        :rtype: str
        """
        return "GraphPattern [{}]".format(str(self))

    def __str__(self) -> str:
        """ Return a string representation

        :rtype: str
        """
        return "{" + "; ".join([str(assertion) for assertion in
                                sorted(self.connections.keys())]) + "}"

    def __hash__(self) -> int:
        return hash(str(self))


class ResourceWrapper(IRIRef):
    """ Resource Wrapper class

    A wrapper which can take on any value of a certain resource, but which
    stores additional information
    """

    def __init__(self, resource:Resource,
                 type:Optional[IRIRef]=None) -> None:
        """ Initialize and instance of this class

        :params resource: the IRI or Literal value 
        :returns: None
        """
        super().__init__(resource)

        self.type = type

    def __eq__(self, other:ResourceWrapper) -> bool:
        if type(other) is not ResourceWrapper:
            return False

        return self.type == other.type\
               and self.value == other.value

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return "RESOURCE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "Resource {} [{}]".format(str(id(self)), str(self))

    def __hash__(self) -> int:
        return hash(str(self))


class Variable(IRIRef):
    """ Variable class

    An unbound variable which can take on any value of a certain object or
    data type resource. Each instance has an unique ID to allow this object to
    be used as variable shared between assertions.
    """

    def __init__(self, resource:IRIRef) -> None:
        """ Initialize and instance of this class

        :params resource: the IRI of the class or datatype 
        :returns: None
        """
        super().__init__(resource)
        self._uuid = uuid4().hex

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return "VAR {}".format(self._uuid)

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "Variable {}".format(self._uuid)

    def __lt__(self, other:Variable) -> bool:
        return str(self) < str(other)

    def __eq__(self, other:Variable) -> bool:
        return type(self) is type(other)\
                and self._uuid == other._uuid

    def equiv(self, other:Variable) -> bool:
        return type(self) is type(other)\
                and self.value == other.value

    def __hash__(self) -> int:
        return hash(str(self))


class TypeVariable(Variable):
    """ Type Variable class

    An unbound variable which can take on any value of a certain object or
    data type resource. Each instance has an unique ID to allow this object to
    be used as variable shared between assertions.
    """

    def __init__(self, resource:IRIRef) -> None:
        """ Initialize and instance of this class

        :params resource: the IRI of the class or datatype 
        :returns: None
        """
        super().__init__(resource)

    def equiv(self, other:TypeVariable) -> bool:
        return type(self) is type(other)\
                and self.value == other.value

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return "TYPE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "TypeVariable {} [{}]".format(self._uuid,
                                             str(self))


class ObjectTypeVariable(TypeVariable):
    """ Object Type Variable class

    An unbound variable which can be any member of an object type class
    (entity)
    """
    def __init__(self, resource:IRIRef):
        """ Initialize and instance of this class

        :params resource: the IRI of the class 
        :returns: None
        """
        super().__init__(resource)

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """

        return "OBJECT TYPE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "ObjectTypeVariable {} [{}]".format(self._uuid,
                                                   str(self))


class DataTypeVariable(TypeVariable):
    """ Data Type Variable class

    An unbound variable which can take on any value of a data type class
    (literal)
    """
    def __init__(self, resource:IRIRef):
        """ Initialize and instance of this class

        :params resource: the IRI of the datatype 
        :returns: None
        """
        super().__init__(resource)

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """

        return "DATA TYPE ({})".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "DataTypeVariable [{}]".format(str(self.value))


class MultiModalVariable(DataTypeVariable):
    """ Multimodal Variable class 

    An unbound variable which conveys a description of a cluster of values for
    node features.
    """
    def __init__(self, resource:IRIRef):
        """ Initialize and instance of this class

        :params resource: the IRI of the datatype 
        :returns: None
        """
        super().__init__(resource)

        self.dtype = resource.value.split('/')[-1]
        if '#' in self.dtype:
            self.dtype = self.dtype.split('#')[-1]

    def equiv(self, other:MultiModalVariable) -> bool:
        return type(self) is type(other)\
                and self.value == other.value\
                and self.dtype == other.dtype

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """

        return "MULTIMODAL [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """

        return "MultiModalVariable [{} - {}]".format(str(self.dtype),
                                                     str(self.value))

class MultiModalNumericVariable(MultiModalVariable):
    """ Numeric Variable class """

    _nds = chr(0x1D4DD)  # normal distribution symbol

    def __init__(self, resource:IRIRef,
                 centroid:tuple[float, float]) -> None:
        """ Initialize and instance of this class

        :returns: None
        """
        super().__init__(resource)
        self.mu, self.var = centroid

    def equiv(self, other:MultiModalNumericVariable) -> bool:
        """ Return true if self and other represent the same
            datatype, and have the same mean and variance.

        :param other:
        :type other: MultiModalNumericVariable
        :rtype: bool
        """
        return type(self) is type(other)\
                and self.dtype == other.dtype\
                and self.value == other.value\
                and self.mu == other.mu\
                and self.var == other.var

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return f"{self.dtype}: {self._nds}({self.mu:.2E}, {self.var:.2E})"

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable

        :rtype: str
        """
        return "MultiModalVariable {}".format(str(self))

class MultiModalStringVariable(MultiModalVariable):
    """ String Variable class """
    def __init__(self, resource, regex) -> None:
        """ Initialize and instance of this class

        :returns: None
        """
        super().__init__(resource)
        self.regex = regex

    def equiv(self, other:MultiModalStringVariable) -> bool:
        """ Return true if self and other represent the same
            datatype, and have the same regular expression.

        :param other:
        :type other: MultiModalStringVariable
        :rtype: bool
        """
        # does not account for equivalent regex patterns
        return type(self) is type(other)\
                and self.value == other.value\
                and self.regex == other.regex

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return f"{self.dtype}: {self.regex}"

    def __repr__(self) -> str:
        """ Return a technical description of this variable

        :returns: a technical description of this variable
        :rtype: str
        """
        return "MultiModalVariable {}".format(str(self))

class Assertion(tuple):
    """ Assertion class

    Wrapper around tuple (an assertion) that gives each instantiation an
    unique uuid which allows for comparisons between assertions with the
    same values. This is needed when either lhs or rhs use TypeVariables.
    """

    def __new__(cls, subject:Union[ResourceWrapper, Variable],
                predicate:Resource,
                object:Union[ResourceWrapper, Variable]) -> Assertion:
        """ Create a new instance of Assertion

        :param subject: the subject of an assertion
        :param predicate: the predicate of an assertion
        :param object: the object of an assertion
        :returns: a new instance of Assertion
        """
        return super().__new__(cls, (subject, predicate, object))

    def __init__(self,  subject:Union[ResourceWrapper, Variable],
                predicate:Resource,
                object:Union[ResourceWrapper, Variable],
                 _uuid:Optional[str] = None) -> None:
        """ Initialize a new instance of Assertion

        :param subject: the subject of an assertion
        :param predicate: the predicate of an assertion
        :param object: the object of an assertion
        :param _uuid: identifier of this instance
        :returns: None
        """
        self.lhs = subject
        self.predicate = predicate
        self.rhs = object

        self.uuid = _uuid if _uuid is not None else 'A' + uuid4().hex
        self._inv_idx_map = dict()  # type: dict[int,set[int]]

    def copy(self, deep:bool = False) -> Assertion:
        """ Return a copy of this object

        :param deep: Copy the UUID and HASH as well
        :returns: a copy of type Assertion
        :rtype: Assertion
        """
        copy = Assertion(self.lhs, self.predicate, self.rhs)
        if deep:
            copy.uuid = self.uuid

        return copy

    def __getnewargs__(self) -> Assertion:
        """ Pickle elements correctly

        :returns: an instance of Assertion
        :rtype: Assertion
        """
        return Assertion(self.lhs, self.predicate, self.rhs)

    def __hash__(self) -> int:
        """ Return unique hash for each assertion, regardless of content.

        :rtype: int
        """
        return hash(str(self))

    def __str__(self) -> str:
        """ Return the string description of this assertion

        :returns: a description of this assertion
        :rtype: str
        """
        return "(" + ', '.join([str(self.lhs),
                                str(self.predicate),
                                str(self.rhs)]) + ")"

    def __lt__(self, other:Assertion) -> bool:
        """ Compare assertions per element

        :returns: true if self < other else false
        :rtype: str
        """
        if self.predicate < other.predicate:
            return True
        if self.predicate == other.predicate:
            if self.lhs < other.lhs:
                return True
            if self.lhs == other.lhs and self.rhs < other.rhs:
                return True

        return False

    def equal(self, other:Assertion) -> bool:
        """ Return true if assertions are equal content wise

        :param other:
        :type other: Assertion
        :rtype: bool
        """
        if self.predicate == other.predicate\
            and self.lhs == self.lhs\
            and self.rhs == self.rhs:
                return True

        return False

    def equiv(self, other:Assertion) -> bool:
        """ Return true is assertions are equivalent, which implies that they
        state the same or have an entity/attribute that is an instance of the
        type specified by the other.

        :param other:
        :type other: Assertion
        :rtype: bool
        """
        if self.equal(other):
            return True

        if self.predicate != other.predicate\
            or not self.lhs.equiv(other.lhs):
                return False

        # one is an instance of the other's type
        for a_rhs, b_rhs in [(self.rhs, other.rhs),
                             (other.rhs, self.rhs)]:
            if isinstance(a_rhs, TypeVariable)\
                and isinstance(b_rhs, ResourceWrapper)\
                and a_rhs.value == b_rhs.type:
                    return True

        return False
        

class GenerationForest():
    """ Generation Forest class

    Contains one or more generation trees (one per entity type) and serves as a
    wrapper for tree operations.
    """

    def __init__(self) -> None:
        """ Initialize an empty generation forest

        :returns: None
        """
        self._trees = dict()
        self.types = list()

    def add(self, class_iri:IRIRef, depth:int, pattern:GraphPattern) -> None:
        """ Add a pattern to the generation tree at a certain depth

        :params class_iri: the class IRI as IRIRef instance
        :params depth: the depth at which to add the pattern
        :params pattern: an instance of the pattern class
        :returns: None
        """
        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].add(pattern, depth)

    def update_tree(self, class_iri:IRIRef,
                    patterns:Union[List[GraphPattern],Set[GraphPattern]],
                    depth:int) -> None:
        """ Add / update a list or set of patterns at a certain depth

        :params class_iri: the class IRI as IRIRef instance
        :param patterns: a list or set of instances of the type pattern
        :param depth: the depth of the tree onto which to add the patterns
        :returns: None
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].update(patterns, depth)

    def get(self, class_iri:Optional[IRIRef] = None,
            depth:Optional[int] = None) -> Iterator[GraphPattern]:
        """ Get all patterns from a tree of a certain type
        
        :param class_iri: the class IRI as IRIRef instance
        :param depth: the depth of the tree from which to return the patterns.
            Returns all patterns if depth is None (default).
        :returns: Iterator over patterns
        """
        if class_iri is None:
            for class_iri in self._trees.keys():
                for pattern in self._trees[class_iri].get(depth):
                    yield pattern
            return

        if class_iri not in self._trees.keys():
            raise KeyError()

        for pattern in self._trees[class_iri].get(depth):
            yield pattern

    def get_tree(self, class_iri:IRIRef) -> GenerationTree:
        """ Get a tree of a certain type
        
        :param class_iri: the class IRI as IRIRef instance
        :returns: an instance of GenerationTree
        :rtype: GenerationTree
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        return self._trees[class_iri]

    def prune(self, class_iri:IRIRef, depth:int,
              patterns:Union[List[GraphPattern],Set[GraphPattern]]) -> None:
        """ Remove a list or set of patterns at a certain depth

        :param class_iri: the class IRI as IRIRef instance
        :param patterns: a list or set of instances of the type pattern
        :returns: None
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].prune(patterns, depth)

    def clear(self, class_iri:IRIRef, depth:int) -> None:
        """ Clear all patterns on a certain depth of a tree

        :param class_iri: the class IRI as IRIRef instance
        :param depth: the depth of the tree from which to remove the patterns
        :returns: None
        """
        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].clear(depth)

    def plant(self, class_iri:IRIRef, tree:GenerationTree) -> None:
        """ Plant a tree of a certain type in the forest

        :param class_iri: the class IRI as IRIRef instance
        :param tree: an instance of GenerationTree
        :returns: None
        """
        if type(tree) is not GenerationTree:
            raise TypeError()

        self._trees[class_iri] = tree
        self.types.append(class_iri)

    def __len__(self) -> int:
        """ Return the number of trees in the forest

        :returns: the size of the forest
        :rtype: int
        """
        return len(self._trees)

    def __str__(self) -> str:
        """ Return a description of the forest

        :returns: a description of the forest
        :rtype: str
        """
        return "; ".join({"{} ({})".format(t, str(self.get_tree(t))) for t in self._trees.keys()})

    def __contains__(self, pattern:GraphPattern) -> bool:
        for tree in self._trees.values():
            if pattern in tree:
                return True

        return False

class GenerationTree():
    """ Generation Tree class

    A multitree consisting of all patterns that hold for entities of a certain
    type t. All patterns of depth 0 (body := {type(e, t)}) form the roots of the
    tree, with each additional depth consisting of one or more constraint
    patterns that expand their parents' body by one assertion.
    """

    def __init__(self, identifier:Optional[Union[str, IRIRef]] = None) -> None:
        """ Initialize an empty generation tree

        :param identifier: an optional identifier
        :returns: None
        """
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = 'T' + uuid4().hex

        self._tree = list()
        self.height = 0  # number of levels
        self.size = 0  # number of vertices

    def add(self, pattern:GraphPattern, depth:int) -> None:
        """ Add a pattern to the tree at a certain depth

        :param pattern: an instance of the type GraphPattern
        :param depth: the depth of the tree onto which to add the pattern
        :returns: None
        """
        if type(pattern) is not GraphPattern:
            raise TypeError(f"Pattern is of wrong type: {type(pattern)}")
        if depth > self.height:
            raise IndexError("Depth exceeds height of tree")
        if self.height <= depth:
            self._tree.append(set())
            self.height += 1

        self._tree[depth].add(pattern)
        self.size += 1

    def rmv(self, pattern:GraphPattern, depth:int) -> None:
        """ Remove a pattern from the tree at a certain depth

        :param pattern: an instance of the type GraphPattern
        :param depth: the depth of the tree from which to remove the pattern
        :returns: None
        """
        if type(pattern) is not GraphPattern:
            raise TypeError(f"Pattern is of wrong type: {type(pattern)}")
        if depth >= self.height:
            raise IndexError("Depth exceeds height of tree")

        self._tree[depth].remove(pattern)
        self.size -= 1

    def update(self, patterns:Union[List[GraphPattern],Set[GraphPattern]],
               depth:int) -> None:
        """ Add / update a list or set of patterns at a certain depth

        :param patterns: a list or set of instances of the type GraphPattern
        :param depth: the depth of the tree onto which to add the patterns
        :returns: None
        """
        # redundancy needed for case if len(patterns) == 0
        if depth > self.height:
            raise IndexError("Depth exceeds height of tree")
        if self.height <= depth:
            self._tree.append(set())
            self.height += 1

        for pattern in patterns:
            self.add(pattern, depth)

    def prune(self, patterns:Union[List[GraphPattern],Set[GraphPattern]],
              depth:int) -> None:
        """ Remove a list or set of patterns at a certain depth

        :param patterns: a list or set of instances of the type GraphPattern
        :param depth: the depth of the tree from which to remove the patterns
        :returns: None
        """
        for pattern in patterns:
            self.rmv(pattern, depth)

    def clear(self, depth:int) -> None:
        """ Clear all patterns on a certain depth
        
        :param depth: the depth of the tree from which to remove the patterns
        :returns: None
        """
        if depth >= self.height:
            raise IndexError("Depth exceeds height of tree")

        self.size -= len(self._tree[depth])
        self._tree[depth] = set()

    def get(self, depth:Optional[int] = None) -> Iterator[GraphPattern]:
        """ Return all graph_patterns from the tree
        
        :param depth: the depth of the tree from which to return the
                      graph_patterns.
                      Returns all graph_patterns if depth is None (default).
        :returns: Iterator over graph_patterns
        :rtype: Iterator[graph_pattern]
        """
        if depth is None:
            if len(self._tree) > 0:
                for graph_pattern in set.union(*self._tree):
                    yield graph_pattern
        else:
            if depth >= self.height:
                raise IndexError("Depth exceeds height of tree")

            for pattern in self._tree[depth]:
                yield pattern

    def __len__(self) -> int:
        """ Return the length of the tree from the roots to the lowest leafs

        :returns: the length of the tree
        :rtype: int
        """
        return self.height

    def __str__(self) -> str:
        """ Return a description of the tree

        :returns: a description of the tree
        :rtype: str
        """
        return "{}:{}:{}".format(self.identifier, self.height, self.size)
    
    def __contains__(self, pattern:GraphPattern) -> bool:
        for layer in self._tree:
            if pattern in layer:
                return True

        return False

