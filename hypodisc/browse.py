#!/usr/bin/env python

import argparse
import codecs
import logging
from os import access, getcwd, R_OK
from os.path import isfile
from threading import Timer
from typing import Generator, Tuple, Union
import webbrowser

from flask import Flask, render_template

from rdf.graph import Statement
from rdf.formats import NTriples
from rdf.namespaces import RDF, RDFS, XSD
from rdf.terms import IRIRef, Literal


REPO_IRI = IRIRef("https://gitlab.com/wxwilcke/hypodisc#")
MLS_IRI = IRIRef("http://www.w3.org/ns/mls#")
DCT_IRI = IRIRef("http://purl.org/dc/terms/")

app = Flask(__name__, root_path=f'{getcwd()}/hypodisc/www/')
app.config['SERVER_NAME'] = '127.0.0.1:5000'

@app.route('/')
def init_viewer():
    return render_template("default.html",
                           filename = gfilename,
                           metadata = gmetadata,
                           queries = gqueries)

def open_browser(port:int) -> None:
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}')

def infer_base_ns(s:IRIRef) -> IRIRef:
    if s.value.startswith('file://') and s.value.endswith('.nt'):
        return s + '#'
    else:
        return IRIRef('')

def process_query(data:list, query_id:int) -> dict:
    query = {"id": query_id}
    for _, p, o in data:
        attr = p.value.split('#')[-1][3:].lower()

        value = o.value
        if o.datatype == XSD + 'nonNegativeInteger':
            value = int(value)
        if attr == 'pattern' and type(value) is str:
            value = codecs.decode(value, 'unicode_escape')
            value = '\n' + value

        query[attr] = value

    return query

def process_metadata(data:list, base:IRIRef, run_id:str) -> dict:
    metadata = {'id': run_id}  # type: dict[str, Union[str,list[str]]]

    base_len = len(base)
    for s, p, o in data:
        if s == base:
            if base.value in o.value:
                # skip hyperparameters
                continue

            p_term = p.value.split('#')[-1] if '#' in p.value\
                     else p.value.split('/')[-1]
            if p_term in metadata.keys():
                # use list for multiple values
                o_value = metadata[p_term]
                if type(o_value) is not list:
                    metadata[p_term] = [o_value]
                
                metadata[p_term].append(o.value)
            else:
                metadata[p_term] = o.value

            continue
        elif base.value in s.value:
            s_term = s.value[base_len:]
            if p == MLS_IRI + 'hasValue':
                metadata[s_term[2:]] = o.value

    return metadata

def parse(data:Generator[Statement, None, None],
          base:IRIRef) -> Tuple[dict[str, str], list[dict]]:
    # assume an ordered set
    queries = list()
    metadata = dict()

    s, p, o = next(data)
    if len(base) <= 0:
        base = infer_base_ns(s)  # type:IRIRef
        if base is None:
            raise ValueError('Expects subject of first line '
                             'to contain base IRI')

    base_len = len(base)
    while True:
        try:
            t = next(data)
            if t[0].value[base_len: base_len+5] == 'Query':
                qdata = [t]
                qdata.extend([next(data) for _ in range(4)])

                query = process_query(qdata, int(t[0].value[base_len+6:]))
                queries.append(query)

                continue
            elif t[2] == MLS_IRI + 'Run':
                qdata = [t]
                qdata.extend([next(data) for _ in range(66)])

                metadata = process_metadata(qdata, t[0],
                                            t[0].value.split('#')[-1])

        except StopIteration:
            break

    return metadata, queries

def setup_logger(verbose:bool) -> None:
    """ Setup logger

    :param verbose: print debug messages
    :type verbose: bool
    :rtype: None
    """
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HypoDisc")
    parser.add_argument("input", help="An N-Triple file containing queries "
                        + "with hypotheses")
    parser.add_argument("--base", help="Provide a custom base namespace",
                        default = '')
    parser.add_argument("--port", help="Change the default port",
                        default = 5000, type = int)
    parser.add_argument("--suppress_browser", help="Suppress the call to the "
                        "browser", action="store_true")
    parser.add_argument("--verbose", "-v", help="Print debug messages and "
                        " warnings", action="store_true")
    args = parser.parse_args()

    setup_logger(args.verbose)

    # validate paths 
    if not isfile(args.input): 
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not access(args.input, R_OK):
        raise PermissionError(f"Input path not readable: {args.input}")
 
    g = NTriples(args.input).parse()

    global gfilename
    global gmetadata 
    global gqueries

    gfilename = args.input
    gmetadata, gqueries = parse(g, IRIRef(args.base))

    if not args.suppress_browser:
        Timer(1, open_browser, [args.port]).start()

    with app.app_context():
        init_viewer()
        app.run(debug = args.verbose, port = args.port)

