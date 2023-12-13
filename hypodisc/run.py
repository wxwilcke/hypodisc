#!/usr/bin/env python

import argparse
import logging
from datetime import datetime
from getpass import getuser
from os import access, getcwd, R_OK, W_OK
from os.path import isdir, isfile
from pathlib import Path
from sys import maxsize
from time import time
from typing import Any

from rdf.formats import NTriples
from rdf.namespaces import RDF, RDFS, XSD
from rdf.terms import IRIRef, Literal

from hypodisc.core.sequential import generate as generate_seq
from hypodisc.core.parallel import generate as generate_mp
from hypodisc.core.utils import (floatProbabilityArg, strNamespaceArg,
                                 integerRangeArg,
                                 read_version, rng_set_seed)
from hypodisc.data.graph import KnowledgeGraph, mkprefixes
from hypodisc.data.utils import mkfile, UnsupportedSerializationFormat


PYPROJECTS_PATH = getcwd() + "/pyproject.toml"
VERSION = read_version(PYPROJECTS_PATH)

OUTPUT_NAME = "out"
OUTPUT_EXT = ".nt"
OUTPUT_LANG = "en"
OUTPUT_TYPE = "http://www.w3.org/ns/mls#Model"
OUTPUT_FORMAT = ("https://www.iana.org/assignments/media-types/"
                 "application/n-triples")
OUTPUT_LABEL = "An automatically generated collection of SPARQL queries"
OUTPUT_DESCRIPTION = ("A collection of SPARQL queries generated "
                      "automatically by HypoDisc: a tool, funded "
                      "by CLARIAH and developed by the DataLegend "
                      "team (Xander Wilcke, Richard Zijdeman, Rick "
                      "Mourits, Auke Rijpma, and Sytze van Herck), "
                      "that can discover novel and "
                      "potentially-interesting graph patterns "
                      "in multimodal knowledge graphs which can be "
                      "used by experts and scholars to form new "
                      "research hypotheses, to support existing "
                      "ones, or to gain insight into their data.")
TASK_DESCRIPTION = ("Discovering potentially interesting graph patterns "
                    "in multimodal knowledge graphs.")
REPO_URL = "https://gitlab.com/wxwilcke/hypodisc"

def write_metadata(f_out:NTriples, graph_label:IRIRef,
                   parameters:dict[str,Any]) -> None:
    BASE = IRIRef(REPO_URL) + '#'
    DCT = IRIRef("http://purl.org/dc/terms/")
    MLS = IRIRef("http://www.w3.org/ns/mls#")

    # output type
    f_out.write((graph_label, RDF+"type", IRIRef(OUTPUT_TYPE)))

    # output format
    output_format = IRIRef(OUTPUT_FORMAT)
    f_out.write((graph_label, DCT+"format", output_format))
    f_out.write((output_format, RDF+"type", DCT+"MediaType"))

    # output description
    description = Literal(OUTPUT_DESCRIPTION, language = OUTPUT_LANG)
    f_out.write((graph_label, DCT+"description", description))

    # output label
    label = Literal(OUTPUT_LABEL, language = OUTPUT_LANG)
    f_out.write((graph_label, RDFS+"label", label))

    # creator
    creator = Literal(f"{getuser().title()}", datatype = XSD+'string')
    f_out.write((graph_label, DCT+"creator", creator))

    # this specific run
    r = int(datetime.timestamp(datetime.now()))
    run = graph_label + f"#R{r}"
    task = BASE + "HypothesisDiscovery"
    implementation = IRIRef(REPO_URL)
    
    f_out.write((run, RDF+"type", MLS+"Run"))
    f_out.write((run, MLS+'hasOutput', graph_label))
    f_out.write((run, MLS+"achieves", task))
    f_out.write((run, MLS+"executes", implementation))

    t_start = Literal(f"{datetime.isoformat(datetime.now())}",
                      datatype = XSD+"dateTime")
    f_out.write((run, DCT+"date", t_start))

    task_description = Literal(TASK_DESCRIPTION, language = OUTPUT_LANG)
    f_out.write((task, RDF+"type", MLS+"Task"))
    f_out.write((task, DCT+"description", task_description))

    f_out.write((implementation, RDF+"type", MLS+"Implementation"))
    version = Literal(f"Version {VERSION}", datatype = XSD+"string")
    f_out.write((implementation, MLS+"ImplementationCharacteristic", version))

    # which input files
    for input_file in parameters['input']:
        p = IRIRef(f"file://{Path(input_file).absolute()}")

        f_out.write((run, MLS+'hasInput', p))
        f_out.write((p, RDF+"type", MLS+"Dataset"))

    # all other hyperparameters
    for param, value in parameters.items():
        if param == 'input' or value is None or not value:
            continue

        # hyperparameters of the model
        base_param = BASE + f"P_{param}"
        f_out.write((implementation, MLS+"hasHyperParameter", base_param))

        dtype = XSD + "string"
        if isinstance(value, float):
            dtype = XSD + "float"
        elif isinstance(value, int):
            dtype = XSD + "nonNegativeInteger"

        # hyperparameters used in this run
        v = Literal(str(value), datatype = dtype)
        param_instance = run + f"P_{param}"
        f_out.write((run, MLS+'hasInput', param_instance))
        f_out.write((param_instance, RDF+"type", MLS+"HyperParameterSetting"))
        f_out.write((param_instance, MLS+"specifiedBy", base_param))
        f_out.write((param_instance, MLS+"hasValue", v))


def setup_logger(verbose:bool) -> None:
    """ Setup logger

    :param verbose: print debug messages
    :type verbose: bool
    :rtype: None
    """
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser(prog="HypoDisc")
    parser.add_argument("-d", "--depth", help="Depths to explore. Takes a "
                        + "range 'from:to', or a shorthand ':to' or 'to' if "
                        + "all depths up to that point are to be considered.",
                        type=integerRangeArg, required=True)
    parser.add_argument("-s", "--min_support", help="Minimal pattern support.",
                        type=int, required=True)
    parser.add_argument("-o", "--output", help="Path to write output to.",
                        type=str, default=getcwd())
    parser.add_argument("input", help="One or more knowledge graphs in " 
                        + "(gzipped) NTriple or NQuad serialization format.",
                        nargs='+')
    parser.add_argument("--max_size", help="Maximum context size",
                        type=int, required=False, default=maxsize)
    parser.add_argument("--max_width", help="Maximum width of shell",
                        type=int, required=False, default=maxsize)
    parser.add_argument("--mode", help="A[box], T[box], or both as"
                        + " candidates to be included in the pattern",
                        choices = ["A", "T", "AT", "TA"],
                        type=str, default="AT")
    parser.add_argument("--multimodal", help="Enable multimodal support",
                        required=False, action='store_true')
    parser.add_argument("--namespace", help="Add a custom prefix:namespace "
                        + "pair to be used in the output. This parameter can "
                        + "be used more than once to provide multiple "
                        + "mappings. Must be provided as 'prefix:namespace', "
                        + "eg 'ex:http://example.org/'.",
                        type=strNamespaceArg, action='append', default=[])
    parser.add_argument("--parallel", help="Speed up the computation by "
                        "distributing the search across multiple CPU cores",
                        action="store_true")
    parser.add_argument("--p_explore", help="Probability of exploring "
                        + "candidate endpoint.", type=floatProbabilityArg,
                        required=False, default=1.0)
    parser.add_argument("--p_extend", help="Probability of extending at "
                        + "candidate endpoint.", type=floatProbabilityArg,
                        required=False, default=1.0)
    parser.add_argument("--dry_run", help="Dry run without saving results.",
                        required=False, action="store_true")
    parser.add_argument("--seed", help="Set the seed for the random number "
                        + "generator.", type=int, required=False,
                        default=None)
    parser.add_argument("--verbose", "-v", help="Print debug messages and "
                        " warnings", action="store_true")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")
    args = parser.parse_args()

    setup_logger(args.verbose)

    # print user-provided arguments
    logging.debug("Arguments: " + "; ".join(["{}: {}".format(k,v)
                  for k,v in vars(args).items()]))

    # initialize random number generator
    rng = rng_set_seed(args.seed)

    if args.min_support <= 1:
        raise Warning("A minimal support less than or equal to 1 is unlikely "
                      "to yield interesting patterns (if any).")

    # validate paths 
    for filename in args.input:
        if not isfile(filename): 
            raise FileNotFoundError(f"Input path not found: {filename}")
        if not access(filename, R_OK):
            raise PermissionError(f"Input path not readable: {filename}")
 
    # load graph(s)
    print(f"importing {len(args.input)} graph(s)...", end=" ")
    kg = KnowledgeGraph(rng)
    kg.parse(args.input)
    print("done")

    # validate paths
    f_out = None
    prefix_map = None
    graph_label = None
    if not args.dry_run:
        if isdir(args.output):
            # create new file with default name
            output_path = mkfile(args.output, OUTPUT_NAME, OUTPUT_EXT)
        else:
            output_path = Path(args.output)
 
        if output_path.suffix != OUTPUT_EXT:
            raise UnsupportedSerializationFormat("Specified output path "
                                                "has unexpected extension: "
                                                 f"{args.ouput.suffix}")

        f_out = NTriples(path = output_path, mode = 'w')
        if not isfile(f_out.path):
            raise FileNotFoundError(f"Output path not found: {args.output}")
        if not access(f_out.path, W_OK):
            raise PermissionError(f"Output path not writable: {args.output}")

        print(f"writing output to {output_path}")

        namespaces = {ns:pf for pf, ns in args.namespace}
        prefix_map = mkprefixes(kg.namespaces,
                                namespaces)

        graph_label = IRIRef("file://" + str(output_path))
        # write metadata to output
        write_metadata(f_out, graph_label, vars(args))

    generate = generate_seq
    if args.parallel:
        generate = generate_mp

    # compute clauses
    generate(rng = rng, kg = kg, depths = args.depth,
             min_support = args.min_support,
             p_explore = args.p_explore,
             p_extend = args.p_extend,
             mode = args.mode,
             max_length = args.max_size,
             max_width = args.max_width,
             multimodal = args.multimodal,
             out_writer = f_out,
             out_prefix_map = prefix_map,
             out_ns = graph_label)
