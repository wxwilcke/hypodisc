#!/usr/bin/env python

import argparse
import csv

import pandas as pd
from rdflib import Graph
from rdflib.term import BNode, Literal, URIRef
from rdflib_hdt import HDTStore


def _solve_literal_allignment(df_triples, node,  node_series,
                              subject_predicate):
    annotation = None
    if node.datatype is not None:
        annotation = node.datatype
    elif node.language is not None:
        annotation = '@' + node.language

    if annotation is not None:
        node_series = node_series[node_series.annotation == str(annotation)]

    index = node_series['index']
    if node_series.shape[0] > 1:
        # more than 1 value with the same datatype (or None)
        # retrieve all objects for this s, p pair and check against node_series
        s_int, p_int = subject_predicate
        objects = df_triples[(df_triples['index_lhs_node'] == s_int) &
                             (df_triples['index_relation'] == p_int)]

        index_set = set(objects['index_rhs_node']) & set(node_series['index'])
        if len(index_set) == 1:
            index = index_set.pop()

    return int(index)


def _node_str(node):
    value = str(node)
    value.strip()

    return value


def _node_type(node):
    if isinstance(node, BNode):
        return "blank_node"
    elif isinstance(node, URIRef):
        return "iri"
    elif isinstance(node, Literal):
        if node.datatype is not None:
            return str(node.datatype)
        elif node.language is not None:
            return '@' + node.language
        else:
            return "none"
    else:
        raise Exception()


def _generate_context(graphs):
    entities = set()
    relations = set()
    datatypes = set()

    for g in graphs:
        for s, p, o in g.triples((None, None, None)):
            s_type = _node_type(s)
            o_type = _node_type(o)

            datatypes.add(s_type)
            datatypes.add(o_type)

            entities.add((_node_str(s), s_type))
            entities.add((_node_str(o), o_type))

            relations.add(_node_str(p))

    i2e = list(entities)
    i2r = list(relations)
    i2d = list(datatypes)

    # ensure deterministic order
    i2e.sort()
    i2r.sort()
    i2d.sort()

    e2i = {e: i for i, e in enumerate(i2e)}
    r2i = {r: i for i, r in enumerate(i2r)}

    triples_int = list()
    for g in graphs:
        for s, p, o in g.triples((None, None, None)):
            triples_int.append([e2i[(_node_str(s), _node_type(s))],
                                r2i[_node_str(p)],
                                e2i[(_node_str(o), _node_type(o))]])

    triples_int = pd.DataFrame(triples_int,
                               columns=["index_lhs_node",
                                        "index_relation",
                                        "index_rhs_node"])

    nodes = [(i, dt, ent) for i, (ent, dt) in enumerate(i2e)]
    nodes = pd.DataFrame(nodes, columns=['index', 'annotation', 'label'])
    relations = pd.DataFrame(enumerate(i2r), columns=['index', 'label'])
    nodetypes = pd.DataFrame(enumerate(i2d), columns=['index', 'annotation'])

    return ((nodes, nodetypes, relations, triples_int), e2i, r2i)


def _generate_link_prediction_mapping_standalone(flags):
    g_list = list()
    g_len_list = list()
    for hdtfile in [flags.train, flags.test, flags.valid]:
        if hdtfile is not None:
            store=HDTStore(hdtfile)
            g = Graph(store=store)

            g_list.append(g)
            g_len_list.append(len(store))
        else:
            g_len_list.append(0)

    data, e2i, r2i = _generate_context(g_list)

    num_facts = sum(g_len_list)
    fact_idc = pd.Series(range(num_facts))
    split_idc = pd.Series([*[0]*g_len_list[0],
                           *[1]*g_len_list[1],
                           *[2]*g_len_list[2]])

    df_splits = pd.concat([pd.Series(fact_idc), pd.Series(split_idc)], axis=1)
    df_splits.columns = ["triple_index", "split_index"]

    return (data, df_splits)


def generate_link_prediction_mapping(flags):
    data = None
    if flags.context is None:
        return _generate_link_prediction_mapping_standalone(flags)
    else:
        # allign indices with established data
        path = flags.context if flags.context.endswith('/')\
            else flags.context + '/'
        df_triples = pd.read_csv(path + "triples.int.csv")
        df_nodes = pd.read_csv(path + "nodes.int.csv", index_col='label')
        df_relations = pd.read_csv(path + "relations.int.csv",
                                   index_col='label')

    # map triples to split index
    p_errors = set()
    fact_idc = list()
    split_idc = list()
    facts2i = {tuple(df_triples.iloc[i]): i for i in range(len(df_triples))}
    for i, hdtfile in enumerate([flags.train, flags.test, flags.valid]):
        if hdtfile is None:
            continue

        g = Graph(store=HDTStore(hdtfile))
        for s, p, o in g.triples((None, None, None)):
            s_int = df_nodes.loc[_node_str(s)]['index']
            try:
                p_int = df_relations.loc[_node_str(p)]['index']
            except KeyError:
                # assume this is a target link not included in the
                # classification dataset
                p_errors.add(p)

                continue

            o_series = df_nodes.loc[_node_str(o)]
            o_int = o_series['index']
            if isinstance(o_series, pd.DataFrame):
                # there are more literals with this value
                o_int = _solve_literal_allignment(df_triples, o, o_series,
                                                  (s_int, p_int))

            try:
                index = facts2i[(s_int, p_int, o_int)]
            except Exception:
                print("Error mapping triple: (%s, %s, %s)" % (s, p, o))
                continue

            fact_idc.append(index)
            split_idc.append(i)

    for p in p_errors:
        print("Not in context: %s" % _node_str(p))

    df_splits = pd.concat([pd.Series(fact_idc), pd.Series(split_idc)], axis=1)
    df_splits.columns = ["triple_index", "split_index"]

    return (data, df_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--context", help="Previously generated CSV "
                        + "directory to align to",
                        default=None)
    parser.add_argument("-o", "--output", help="Output directory", default="./")
    parser.add_argument("-ts", "--train", help="Training set (HDT graph)",
                        default=None)
    parser.add_argument("-ws", "--test", help="Withheld set for testing "
                        + "(HDT graph)", default=None)
    parser.add_argument("-vs", "--valid", help="Validation set (HDT graph)",
                        default=None)
    flags = parser.parse_args()

    path = flags.Output if flags.output.endswith('/') else flags.output + '/'

    data, df_splits = generate_link_prediction_mapping(flags)
    df_splits.to_csv(path+'linkprediction_splits.int.csv',
                     index=False, header=True)

    if data is not None:
        df_nodes, df_nodetypes, df_relations, df_triples = data

        df_nodes.to_csv(path+'nodes.int.csv', index=False, header=True,
                        quoting=csv.QUOTE_NONNUMERIC)
        df_relations.to_csv(path+'relations.int.csv', index=False, header=True,
                            quoting=csv.QUOTE_NONNUMERIC)
        df_nodetypes.to_csv(path+'nodetypes.int.csv', index=False, header=True)
        df_triples.to_csv(path+'triples.int.csv', index=False, header=True)
