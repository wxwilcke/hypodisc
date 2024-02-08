#! /usr/bin/env python

from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Set, Union
from uuid import uuid4

from rdf.terms import IRIRef, Resource

from hypodisc.data.graph import ns2pf


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

        if len(assertions) > 0:
            self.root = self._infer_root(assertions)
            self.distances = self._compute_distances({self.root}, assertions)
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
        distance = 0
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
            for d, assertions in self.distances.items():
                if assertion in assertions:
                    distance = d + 1

                    break

        self.connections[extension] = set()
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

    def as_query(self, prefix_map:dict[str, str]) -> str:
        """ Return a SPARQL query representation of this pattern.

        :param prefix_map:
        :type prefix_map: dict[str, str]
        :rtype: str
        """
        q = ''
        
        for ns in sorted(prefix_map.keys()):
            q += fr"PREFIX {prefix_map[ns]}: <{ns}>\n"

        if len(prefix_map) > 0:
            q += r'\n'

        var_i = 97  # start with 'a'

        q += r"SELECT ?s\n"
        q += r"WHERE {\n"

        bindings = dict()
        postpone = dict()
        filter = set()
        for d in self.distances.keys():
            for a in self.distances[d]:
                if isinstance(a.rhs, ObjectTypeVariable):
                    if a.rhs not in bindings.keys():
                        bindings[a.rhs] = "_" + chr(var_i)  # blank node
                        postpone[a.rhs] = True

                        var_i += 1

                q += r'\t'
                if d == 0:
                    q += "?s"
                else:
                    if a.lhs in bindings.keys():  # ObjectTypeVariable
                        if a.lhs in postpone.keys():
                            pfe = ns2pf(prefix_map, a.lhs.value)
                            q += ( f"?{bindings[a.lhs]} rdf:type "
                                   fr"{pfe} .\n\t" )
                            postpone[a.lhs] = False

                        q += f"?{bindings[a.lhs]}"
                    else:
                        q += f"{a.lhs}"

                pfe = ns2pf(prefix_map, a.predicate)
                q += f" {pfe} "
        
                if a.rhs in bindings.keys():  # ObjectTypeVariable
                    q += f"?{bindings[a.rhs]}"
                elif type(a.rhs) is DataTypeVariable:
                    var = '_' + chr(var_i)
                    if type(a.rhs.value) is IRIRef:
                        # data type
                        pfe = ns2pf(prefix_map, a.rhs.value)
                        filter.add(f"DATATYPE(?{var}) == {pfe}")
                    else:  # language tag
                        filter.add(fr"LANG(?{var}) == \"{a.rhs.value}\"")
                    q += f"?{var}"
                    var_i += 1
                elif type(a.rhs) is MultiModalStringVariable:
                    var = '_' + chr(var_i)
                    filter.add(fr"REGEX(STR(?{var}), \"{a.rhs.regex}\")")
                    q += f"?{var}"
                    var_i += 1
                elif type(a.rhs) is MultiModalNumericVariable:
                    pfe = ns2pf(prefix_map, a.rhs.value)
                    var = '_' + chr(var_i)
                    if a.rhs.lower_bound == a.rhs.upper_bound:
                        f = ( fr"\"{a.rhs.lower_bound}\"^^{pfe} == "
                              f"?{var}" )
                    else:
                        f = ( fr"\"{a.rhs.lower_bound}\"^^{pfe} <= "
                              fr"?{var} && ?{var} <= \"{a.rhs.upper_bound}\""
                              f"^^{pfe}")

                    filter.add(f)
                    q += f"?{var}"
                    var_i += 1
                elif type(a.rhs) is ResourceWrapper:
                    if type(a.rhs.value) is IRIRef:
                        pfe = ns2pf(prefix_map, a.rhs.value)
                        q += f"{pfe}"
                    else:
                        q += fr"\"{a.rhs.value}\""
                        if type(a.rhs.type) is IRIRef:
                            # data type
                            pfe = ns2pf(prefix_map, a.rhs.type)
                            q += f"^^{pfe}"
                        else:
                            # language tag
                            q += f"@{a.rhs.type}"
                else:
                    q += f"{a.rhs}"

                q += r' .\n'

            if d < self.depth():
                q += r'\n'

        add_whitespace = True
        for otype in postpone.keys():
            if postpone[otype]:
                if add_whitespace:
                    q += r'\n'

                    add_whitespace = False

                pfe = ns2pf(prefix_map, otype.value)
                q += ( fr"\t?{bindings[otype]} rdf:type "
                       fr"{pfe} .\n" )
                postpone[otype] = False

        if len(filter) > 0:
            q += r"\n\tFILTER (\n\t\t" + r" &&\n\t\t".join(filter) + r"\n\t)\n"

        q += "}"

        return q

    def as_dot(self, prefix_map:dict[str, str]) -> str:
        """ Return a dot representation of this pattern.

        :param prefix_map:
        :type prefix_map: dict[str, str]
        :rtype: str
        """
        labels = dict()
        annotations = set()
        postpone = set()
        entities = dict()
        classes = dict()
        literals = dict()

        i = 1
        var_i = 97  # start with 'a'

        dot = r"strict digraph { "
        dot += r"subgraph glabel { glabel [shape = box,  label = \""
        for k,v in prefix_map.items():
            dot += fr"{v}: <{k}>\l"
        dot += r"\"] } "
        dot += r"edge [ minlen = 2 ];"
        dot += r"\"root\" [shape = oval, label = \"?s\", style = bold ]; "
        for d in self.distances.keys():
            for a in self.distances[d]:
                edge = ''
                if a.lhs == self.root:
                    edge += r"\"root\""
                else:  # ObjectTypeVariable
                    pfe = ns2pf(prefix_map, a.lhs.value)
                    class_lab = fr"\"{pfe}\""

                    if a.lhs.value not in entities.keys():
                        entities[a.lhs.value] = f"n{i}"
                        i += 1

                    node_id = entities[a.lhs.value]
                    edge += node_id

                    if node_id not in labels.keys():
                        labels[node_id] = fr"\"?_{chr(var_i)}\""
                        var_i += 1

                    if class_lab not in classes.keys():
                        classes[class_lab] = f"n{i}"
                        i += 1

                    class_id = classes[class_lab]
                    if class_id not in labels.keys():
                        labels[class_id] = class_lab
                        postpone.add(fr"{node_id} -> {class_id} "
                                     r"[label = \"rdf:type\"]; ")

                edge += " -> "
                if isinstance(a.rhs, ObjectTypeVariable):
                    pfe = ns2pf(prefix_map, a.rhs.value)
                    class_lab = fr"\"{pfe}\""

                    if a.rhs.value not in entities.keys():
                        entities[a.rhs.value] = f"n{i}"
                        i += 1

                    node_id = entities[a.rhs.value]
                    edge += node_id

                    if node_id not in labels.keys():
                        labels[node_id] = fr"\"?_{chr(var_i)}\""
                        var_i += 1

                    if class_lab not in classes.keys():
                        classes[class_lab] = f"n{i}"
                        i += 1

                    class_id = classes[class_lab]
                    if class_id not in labels.keys():
                        labels[class_id] = class_lab
                        postpone.add(fr"{node_id} -> {class_id} "
                                     r"[label = \"rdf:type\"]; ")
                elif type(a.rhs) is DataTypeVariable:
                    literals[a.rhs] = f"n{i}"
                    i += 1

                    node_id = literals[a.rhs]
                    edge += node_id

                    if node_id not in labels.keys():
                        labels[node_id] = fr"\"?_{chr(var_i)}\""
                        var_i += 1

                    ann_id = f"n{i}"
                    i += 1

                    annotations.add(ann_id)
                    if type(a.rhs.value) is IRIRef:
                        # data type
                        pfe = ns2pf(prefix_map, a.rhs.value)
                        labels[ann_id] = fr"\"{pfe}\""
                        postpone.add(fr"{node_id} -> {ann_id} "
                                     "[label = datatype, style = dashed, "
                                     "arrowhead = none]; ")
                    else:  # language tag
                        labels[ann_id] = fr"\"{a.rhs.value}\""
                        postpone.add(fr"{node_id} -> {ann_id} "
                                     "[label = language, style = dashed, "
                                     "arrowhead = none]; ")
                elif type(a.rhs) is MultiModalStringVariable:
                    literals[a.rhs] = f"n{i}"
                    i += 1

                    node_id = literals[a.rhs]
                    edge += node_id

                    if node_id not in labels.keys():
                        labels[node_id] = fr"\"?_{chr(var_i)}\""
                        var_i += 1

                    ann_id = f"n{i}"
                    i += 1

                    annotations.add(ann_id)
                    labels[ann_id] = fr"\"\\\"{a.rhs.regex}\\\"\""
                    postpone.add(fr"{node_id} -> {ann_id} "
                                 "[label = regex, style = dashed, "
                                 "arrowhead = none]; ")
                elif type(a.rhs) is MultiModalNumericVariable:
                    literals[a.rhs] = f"n{i}"
                    i += 1

                    node_id = literals[a.rhs]
                    edge += node_id

                    if node_id not in labels.keys():
                        labels[node_id] = fr"\"?_{chr(var_i)}\""
                        var_i += 1

                    ann_id = f"n{i}"
                    i += 1

                    annotations.add(ann_id)

                    pfe = ns2pf(prefix_map, a.rhs.value)
                    labels[ann_id] = ( fr"\"{pfe}\"")
                    postpone.add(fr"{node_id} -> {ann_id} "
                                 "[label = datatype, style = dashed, "
                                 "arrowhead = none]; ")

                    ann_id = f"n{i}"
                    i += 1

                    annotations.add(ann_id)

                    labels[ann_id] = ( fr"\"[{a.rhs.lower_bound}, "
                                        fr"{a.rhs.upper_bound}]\"")
                    postpone.add(fr"{node_id} -> {ann_id} "
                                 "[label = range, style = dashed, "
                                 "arrowhead = none]; ")

                elif type(a.rhs) is ResourceWrapper:
                    if type(a.rhs.value) is IRIRef:
                        if a.rhs.value not in entities.keys():
                            entities[a.rhs.value] = f"n{i}"
                            i += 1

                        node_id = entities[a.rhs.value]
                        edge += node_id

                        pfe = ns2pf(prefix_map, a.rhs.value)
                        labels[node_id] = fr"\"{pfe}\""
                    else:
                        literals[a.rhs] = f"n{i}"
                        i += 1

                        node_id = literals[a.rhs]
                        edge += node_id
                            
                        ann_id = f"n{i}"
                        i += 1
                        
                        annotations.add(ann_id)
                        if type(a.rhs.type) is IRIRef:
                            # data type
                            pfe = ns2pf(prefix_map, a.rhs.type)
                            labels[node_id] = fr"\"{a.rhs.value}\""

                            labels[ann_id] = fr"\"{pfe}\""
                            postpone.add(fr"{node_id} -> {ann_id} "
                                         "[label = datatype, style = dashed, "
                                         "arrowhead = none]; ")
                        else:
                            # language tag
                            labels[node_id] = fr"\"{a.rhs.value}\""

                            labels[ann_id] = fr"\"{a.rhs.type}\""
                            postpone.add(fr"{node_id} -> {ann_id} "
                                         "[label = language, style = dashed, "
                                         "arrowhead = none]; ")

                pfe = ns2pf(prefix_map, a.predicate)
                edge += fr"[label = \"{pfe}\"]; "

                dot += edge

                for t in postpone:
                    dot += t

        for node_id in entities.values():
            lab = labels[node_id]
            dot += fr"{node_id} [shape = oval, label = {lab}]; "

        for node_id in classes.values():
            lab = labels[node_id]
            dot += fr"{node_id} [shape = box, label = {lab}]; "

        for node_id in literals.values():
            lab = labels[node_id]
            dot += fr"{node_id} [shape = plaintext, label = {lab}]; "

        for node_id in annotations:
            lab = labels[node_id]
            dot += fr"{node_id} [shape = plain, label = {lab}]; "

        return dot + '}'

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

    _leq = chr(0x2264)  # <= sign
    def __init__(self, resource:IRIRef,
                 crange:tuple[float, float]) -> None:
        """ Initialize and instance of this class

        :returns: None
        """
        super().__init__(resource)
        self.lower_bound, self.upper_bound = crange

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
                and self.lower_bound == other.lower_bound\
                and self.upper_bound == other.upper_bound

    def __str__(self) -> str:
        """ Return a description of this variable

        :returns: a description of this variable
        :rtype: str
        """
        return (f"{self.dtype}: {self.lower_bound:.2E} "
                f"{self._leq} x {self._leq} {self.upper_bound:.2E}")

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
