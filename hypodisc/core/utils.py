#! /usr/bin/env python

import tomllib
from typing import Optional, Tuple
import random

import numpy as np

from rdf.terms import Literal, IRIRef
from rdf.namespaces import RDF, RDFS, XSD

from hypodisc.core.structures import Assertion, GraphPattern, Variable


def predict_hash(pattern:GraphPattern, endpoint:Variable,
                 extension:Assertion) -> int:
    connections = {Assertion(endpoint,
                             extension.predicate,
                             extension.rhs)}
    connections.update(pattern.connections.keys())

    return hash("{"
                + "; ".join([str(assertion)
                             for assertion in sorted(connections)])
                + "}")

def floatProbabilityArg(arg:str) -> float:
    """ Custom argument type for probability

    :param arg: user provided argument string containing a real-valued value
    :type arg: str
    :rtype: float
    :returns: a probability in [0, 1]
    """
    p = 1.0
    try:
        p = float(arg)
    except:
        raise Exception(f"'{arg}' is not a valid probability. "
                        + "Expects a real value in [0, 1].")

    assert 0. <= p <= 1.

    return p

def strNamespaceArg(arg:str) -> Tuple[str, str]:
    try:
        arg_split = arg.split(':')

        pf = arg_split[0]
        ns = ':'.join(arg_split[1:])

        if ns.startswith('<') and ns.endswith('>'):
            ns = ns[1:-1]
    except:
        raise Exception(f"'{arg}' is not a valid list of "
                        +"prefix:namespace pairs.")

    return pf, ns

def integerRangeArg(arg:str) -> range:
    """ Custom argument type for range

    :param arg: user provided argument string of form 'from:to', ':to', or
        'to', with 'from' and 'to' being positive integers.
    :type arg: str
    :rtype: range
    :returns: range of values to explore
    """

    begin = 0
    arg_lst = arg.split(':')
    try:
        end = int(arg_lst[-1])
        if len(arg_lst) > 1 and len(arg_lst[0]) > 0:
            begin = int(arg_lst[0])
    except:
        raise Exception("'" + arg + "' is not a range of numbers. "
                        + "Expects '0:3', ':3', or '3'.")

    # check if range is valid
    assert begin >= 0 and end >= 0 and begin <= end

    return range(begin, end)

def read_version(filename:str) -> str:
    """ Parse the project's version

    :param filename: path to 'pyproject.toml'
    :type filename: str
    :rtype: str
    :returns: the project's version as a string
    """
    with open(filename, 'rb') as f:
        rc = tomllib.load(f)
    try:
        version = rc["project"]["version"]
    except:
        version = "unknown"

    return version

def rng_set_seed(seed:Optional[int] = None) -> np.random.Generator:
    """ Set seed of the random number generators.

    :param seed: a custom seed (optional)
    :type seed: Optional[int]
    :rtype: np.random.Generator
    :returns: a random number generator
    """
    random.seed(a = seed)  # set Python seed

    return np.random.default_rng(seed)
