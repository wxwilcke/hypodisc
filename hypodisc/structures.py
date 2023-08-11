#! /usr/bin/env python

from __future__ import annotations
from re import fullmatch
from typing import Dict, Iterator, List, Optional, Set, Union
from uuid import uuid4

from rdf.terms import IRIRef, Resource


class Clause():
    """ Clause class

    A clause consists of a head (Assertion) which holds, with probability Pd and
    Pr, for all members of a type t if these members satisfy the constraints in
    the body. Keeps track of its parent and of all members it satisfies
    for efficient computation of support and confidence.
    """

    def __init__(self, head:Assertion, body:ClauseBody,
                 probability:float,
                 confidence:int, support:int,
                 satisfy_complete:set, 
                 satisfy_body:set, 
                 parent:Optional[Clause]) -> None:
        """ Initialize a new instance of the Clause class

        :param head: an Assertion (lhs, pred, rhs) with lhs a variable, pred an
            IRIRef, and rhs a IRIRef, Literal, or variable.
        :param body: an instance of ClauseBody; one or more generalized
            assertions that must all be true for the head to be true as well
        :param probability: probability that an arbitrary member of the
            domain satisfies the head
        :param confidence: number of members in the domain who satisfy both
            body and head
        :param support: number of members in the domain who satisfy the body
        :param satisfy_complete: indices of members in the domain who satisfy
            both body and head
        :param satisfy_body: indices of members in the domain who satisfy the
            body
        :param parent: a instance of the Claus type with a less constrained
            body and an equal or higher confidence and support
        :returns: None
        """
        self.head = head 
        self.body = body
        self.probability = probability
        self.confidence = confidence
        self.support = support
        self.parent = parent

        #self.children = set()
        self.satisfy_body = satisfy_body  # type:set
        self.satisfy_complete = satisfy_complete  # type:set

    def __len__(self) -> int:
        """ Length of the clause measures as the number of assertions in the
            body

        :returns: length as an int
        """
        return len(self.body)

    def width(self, depth:Optional[int]=None) -> int:
        """Return the maximum number of assertions on a certain depth in the
        body, or the most overall if depth is None.

        :rtype: int
        """
        return self.body.width(depth)

    def __lt__(self, other:Clause) -> bool:
        """ Comparison between class instances

        :param other: instance of the Clause class
        :returns: true if self is less than other else false
        """
        if self.body < other.body:
            return True
        if self.head < other.head:
            return True

        return False

    def __str__(self) -> str:
        """ Return a description of this clause

        :returns: a description of this clause
        """
        return "[P:{:0.3f}, Supp:{}, Conf:{}] {} <- {{{}}}".format(
            self.probability,
            self.support,
            self.confidence,
            str(self.head),
            str(self.body))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "Clause [{}]".format(str(self))


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

    def __str__(self) -> str:
        """ Return a description of this clause

        :returns: a description of this clause
        """
        return "RESOURCE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "Resource {} [{}]".format(str(id(self)), str(self))


class TypeVariable(IRIRef):
    """ Type Variable class

    An unbound variable which can take on any value of a certain object or
    data type resource
    """

    def __init__(self, resource:IRIRef) -> None:
        """ Initialize and instance of this class

        :params resource: the IRI of the class or datatype 
        :returns: None
        """
        super().__init__(resource)

    #def __eq__(self, other):
    #    return type(self) is type(other)\
    #            and self.type is other.type

    #def __lt__(self, other):
    #    return self.type < other.type

    #def __hash__(self):
    #    return hash(str(self.__class__.__name__)+str(self.type))

    def __str__(self) -> str:
        """ Return a description of this clause

        :returns: a description of this clause
        """
        return "TYPE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "TypeVariable {} [{}]".format(str(id(self)),
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
        """ Return a description of this clause

        :returns: a description of this clause
        """

        return "OBJECT TYPE [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "ObjectTypeVariable {} [{}]".format(str(id(self)),
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
        """ Return a description of this clause

        :returns: a description of this clause
        """

        return "DATA TYPE ({})".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "DataTypeVariable [{}]".format(str(self))


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

    def __str__(self) -> str:
        """ Return a description of this clause

        :returns: a description of this clause
        """

        return "MULTIMODAL [{}]".format(str(self.value))

    def __repr__(self) -> str:
        """ Return a technical description of this clause

        :returns: a technical description of this clause
        """

        return "MultiModalVariable [{}]".format(str(self))

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

    def __eq__(self, other:MultiModalNumericVariable) -> bool:
        return type(self) is type(other)\
                and self.value is other.value\
                and self.mu == other.mu\
                and self.var == other.var

    def __lt__(self, other:MultiModalNumericVariable) -> bool:
        return type(self) is type(other)\
                and (self.mu < other.mu\
                    or (self.mu == other.mu\
                        and self.var < other.var))

    def __str__(self) -> str:
        return f"{self.dtype}: {self._nds}({self.mu:.2E}, {self.var:.2E})"

    def __repr__(self) -> str:
        return "MultiModalVariable {}".format(str(self))

class MultiModalStringVariable(MultiModalVariable):
    """ String Variable class """
    regex = ""

    def __init__(self, resource, regex) -> None:
        """ Initialize and instance of this class

        :returns: None
        """
        super().__init__(resource)
        self.regex = regex

    def contains(self, s:str) -> bool:
        return fullmatch(self.regex, s) is not None

    def __eq__(self, other:MultiModalStringVariable) -> bool:
        # does not account for equivalent regex patterns
        return type(self) is type(other)\
                and self.value is other.value\
                and self.regex == other.regex

    def __lt__(self, other:MultiModalStringVariable) -> bool:
        # this should idealy be that one regex pattern is less constrained than
        # the other.
        return self.regex < other.regex

    def __str__(self) -> str:
        return f"{self.dtype}: {self.regex}"

    def __repr__(self) -> str:
        return "MultiModalVariable {}".format(str(self))

class Assertion(tuple):
    """ Assertion class

    Wrapper around tuple (an assertion) that gives each instantiation an
    unique uuid which allows for comparisons between assertions with the
    same values. This is needed when either lhs or rhs use TypeVariables.
    """

    def __new__(cls, subject:Union[ResourceWrapper, TypeVariable],
                predicate:Resource,
                object:Union[ResourceWrapper, TypeVariable]) -> Assertion:
        """ Create a new instance of Assertion

        :param subject: the subject of an assertion
        :param predicate: the predicate of an assertion
        :param object: the object of an assertion
        :returns: a new instance of Assertion
        """
        return super().__new__(cls, (subject, predicate, object))

    def __init__(self,  subject:Union[ResourceWrapper, TypeVariable],
                predicate:Resource,
                object:Union[ResourceWrapper, TypeVariable],
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

    def copy(self, deep:bool = False) -> Assertion:
        """ Return a copy of this object

        :param deep: Copy the UUID and HASH as well
        :returns: a copy of type Assertion
        """
        copy = Assertion(self.lhs, self.predicate, self.rhs)
        if deep:
            copy.uuid = self.uuid

        return copy

    def __getnewargs__(self) -> Assertion:
        """ Pickle elements correctly

        :returns: an instance of Assertion
        """
        return Assertion(self.lhs, self.predicate, self.rhs)

    def __hash__(self) -> int:
        """ Return unique hash for each assertion, regardless of content.

        :rtype: str
        """
        return hash(self.uuid)

    def __str__(self) -> str:
        """ Return the string description of this assertion

        :returns: a description of this assertion
        """
        return "(" + ', '.join([str(self.lhs),
                                str(self.predicate),
                                str(self.rhs)]) + ")"

    def __lt__(self, other:Assertion) -> bool:
        """ Compare assertions per element

        :returns: true if self < other else false
        """
        for a, b in [(self.lhs, other.lhs),
                     (self.predicate, other.predicate),
                     (self.rhs, other.rhs)]:
            if a < b:
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

        # one is an instance of the other's type
        for a_rhs, b_rhs in [(self.rhs, other.rhs),
                             (other.rhs, self.rhs)]:
            if isinstance(a_rhs, TypeVariable)\
                and isinstance(b_rhs, ResourceWrapper)\
                and a_rhs.value == b_rhs.type:
                    return True

        return False
        

class IdentityAssertion(Assertion):
    """ Identity Assertion class

    Special class for identity assertion to allow for each recognition and
    selection.
    """
    def __new__(cls, subject:Resource, predicate:Resource,
                object:Resource) -> Assertion:
        """ Create a new instance of IdentityAssertion

        :param subject: the subject of an assertion
        :param predicate: the predicate of an assertion
        :param object: the object of an assertion
        :returns: a new instance of IdentityAssertion
        """
        return super().__new__(cls, subject, predicate, object)

    def copy(self, deep:bool = False) -> IdentityAssertion:
        """ Return a copy of this object

        :param deep: Copy the UUID and HASH as well
        :returns: a copy of type IdentityAssertion
        """
        copy = IdentityAssertion(self.lhs, self.predicate, self.rhs)
        if deep:
            copy.uuid = self.uuid

        return copy

class ClauseBody():
    """ Clause Body class

    Holds all assertions of a clause's body and keeps
    track of the connections and distances (from the root) of these assertions.
    """

    def __init__(self, identity:IdentityAssertion) -> None:
        """__init__.

        :param identity:
        :type identity: IdentityAssertion
        :rtype: None
        """
        self.identity = identity

        self.connections = {identity: set()}  # type: Dict[Assertion, set]
        self.distances = {0: {identity}}  # type: Dict[int, set]
        self.distances_reverse = {identity: 0}  # type: Dict[Assertion, int]

    def extend(self, endpoint:Assertion, extension:Assertion) -> None:
        """ Extend the body by appending a new assertion to an existing
        endpoint.

        :param endpoint:
        :type endpoint: Assertion
        :param extension:
        :type extension: Assertion
        :rtype: None
        """
        assert endpoint in self.connections.keys()

        self.connections[endpoint].add(extension)
        self.connections[extension] = set()

        distance = self.distances_reverse[endpoint] + 1
        self.distances_reverse[extension] = distance
        if distance not in self.distances.keys():
            self.distances[distance] = set()
        self.distances[distance].add(extension)

    def copy(self) -> ClauseBody:
        """ Create a deep copy, except for the assertions, which remain
        as pointers.

        :rtype: ClauseBody
        """
        c = ClauseBody(self.identity)
        c.connections = {k: {v for v in self.connections[k]}
                         for k in self.connections.keys()}
        c.distances = {k: {v for v in self.distances[k]}
                       for k in self.distances.keys()}
        c.distances_reverse = {k: v for k,v in self.distances_reverse.items()}

        return c

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
        return "BODY [{}]".format(str(self))

    def __str__(self) -> str:
        """ Return a string representation

        :rtype: str
        """
        return "{" + "; ".join([str(assertion) for connections in
                                sorted(self.connections.values())
                                for assertion in connections]) + "}"


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

    def add(self, class_iri:IRIRef, depth:int, clause:Clause) -> None:
        """ Add a clause to the generation tree at a certain depth

        :params class_iri: the class IRI as IRIRef instance
        :params depth: the depth at which to add the clause
        :params clause: an instance of the Clause class
        :returns: None
        """
        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].add(clause, depth)

    def update_tree(self, class_iri:IRIRef,
                    clauses:Union[List[Clause],Set[Clause]],
                    depth:int) -> None:
        """ Add / update a list or set of clauses at a certain depth

        :params class_iri: the class IRI as IRIRef instance
        :param clauses: a list or set of instances of the type Clause
        :param depth: the depth of the tree onto which to add the clauses
        :returns: None
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].update(clauses, depth)

    def get(self, class_iri:Optional[IRIRef],
            depth:Optional[int] = None) -> Iterator[Clause]:
        """ Get all clauses from a tree of a certain type
        
        :param class_iri: the class IRI as IRIRef instance
        :param depth: the depth of the tree from which to return the clauses.
            Returns all clauses if depth is None (default).
        :returns: Iterator over clauses
        """
        if class_iri is None:
            for class_iri in self._trees.keys():
                for clause in self._trees[class_iri].get(depth):
                    yield clause
            return

        if class_iri not in self._trees.keys():
            raise KeyError()

        return self._trees[class_iri].get(depth)

    def get_tree(self, class_iri:IRIRef) -> GenerationTree:
        """ Get a tree of a certain type
        
        :param class_iri: the class IRI as IRIRef instance
        :returns: an instance of GenerationTree
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        return self._trees[class_iri]

    def prune(self, class_iri:IRIRef, depth:int,
              clauses:Union[List[Clause],Set[Clause]]) -> None:
        """ Remove a list or set of clauses at a certain depth

        :param class_iri: the class IRI as IRIRef instance
        :param clauses: a list or set of instances of the type Clause
        :returns: None
        """

        if class_iri not in self._trees.keys():
            raise KeyError()

        self._trees[class_iri].prune(clauses, depth)

    def clear(self, class_iri:IRIRef, depth:int) -> None:
        """ Clear all clauses on a certain depth of a tree

        :param class_iri: the class IRI as IRIRef instance
        :param depth: the depth of the tree from which to remove the clauses
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
        """
        return len(self._trees)

    def __str__(self) -> str:
        """ Return a description of the forest

        :returns: a description of the forest
        """
        return "; ".join({"{} ({})".format(t, str(self.get_tree(t))) for t in self._trees.keys()})


class GenerationTree():
    """ Generation Tree class

    A multitree consisting of all clauses that hold for entities of a certain
    type t. All clauses of depth 0 (body := {type(e, t)}) form the roots of the
    tree, with each additional depth consisting of one or more constraint
    clauses that expand their parents' body by one assertion.
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

    def add(self, clause:Clause, depth:int) -> None:
        """ Add a clause to the tree at a certain depth

        :param clause: an instance of the type Clause
        :param depth: the depth of the tree onto which to add the clause
        :returns: None
        """
        if type(clause) is not Clause:
            raise TypeError()
        if depth > self.height:
            raise IndexError("Depth exceeds height of tree")
        if self.height <= depth:
            self._tree.append(set())
            self.height += 1

        self._tree[depth].add(clause)
        self.size += 1

    def rmv(self, clause:Clause, depth:int) -> None:
        """ Remove a clause from the tree at a certain depth

        :param clause: an instance of the type Clause
        :param depth: the depth of the tree from which to remove the clause
        :returns: None
        """
        if type(clause) is not Clause:
            raise TypeError()
        if depth >= self.height:
            raise IndexError("Depth exceeds height of tree")

        self._tree[depth].remove(clause)
        self.size -= 1

    def update(self, clauses:Union[List[Clause],Set[Clause]],
               depth:int) -> None:
        """ Add / update a list or set of clauses at a certain depth

        :param clauses: a list or set of instances of the type Clause
        :param depth: the depth of the tree onto which to add the clauses
        :returns: None
        """
        # redundancy needed for case if len(clauses) == 0
        if depth > self.height:
            raise IndexError("Depth exceeds height of tree")
        if self.height <= depth:
            self._tree.append(set())
            self.height += 1

        for clause in clauses:
            self.add(clause, depth)

    def prune(self, clauses:Union[List[Clause],Set[Clause]],
              depth:int) -> None:
        """ Remove a list or set of clauses at a certain depth

        :param clauses: a list or set of instances of the type Clause
        :param depth: the depth of the tree from which to remove the clauses
        :returns: None
        """
        for clause in clauses:
            self.rmv(clause, depth)

    def clear(self, depth:int) -> None:
        """ Clear all clauses on a certain depth
        
        :param depth: the depth of the tree from which to remove the clauses
        :returns: None
        """
        if depth >= self.height:
            raise IndexError("Depth exceeds height of tree")

        self.size -= len(self._tree[depth])
        self._tree[depth] = set()

    def get(self, depth:Optional[int] = None) -> Iterator[Clause]:
        """ Return all clauses from the tree
        
        :param depth: the depth of the tree from which to return the clauses.
            Returns all clauses if depth is None (default).
        :returns: Iterator over clauses
        """
        if depth is None:
            if len(self._tree) > 0:
                for clause in set.union(*self._tree):
                    yield clause
        else:
            if depth >= self.height:
                raise IndexError("Depth exceeds height of tree")

            for clause in self._tree[depth]:
                yield clause

    def __len__(self) -> int:
        """ Return the length of the tree from the roots to the lowest leafs

        :returns: the length of the tree
        """
        return self.height

    def __str__(self) -> str:
        """ Return a description of the tree

        :returns: a description of the tree
        """
        return "{}:{}:{}".format(self.identifier, self.height, self.size)
