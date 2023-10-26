#!/usr/bin/env python

import argparse
import logging
from os import access, getcwd, R_OK, W_OK
from os.path import isdir, isfile
from pathlib import Path
from sys import maxsize, exit
from time import time

from core.sequential import generate
from core.utils import (floatProbabilityArg, strNamespaceArg, integerRangeArg,
                        read_version, rng_set_seed)
from data.graph import KnowledgeGraph, mkprefixes
from data.json import JSONStreamer, write_context, write_metadata, write_query
from data.utils import mkfile, UnsupportedSerializationFormat


PYPROJECTS_PATH = getcwd() + "/pyproject.toml"
VERSION = read_version(PYPROJECTS_PATH)
OUTPUT_NAME = "out"
OUTPUT_EXT = ".jsonld"

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
                        type=strNamespaceArg, action='append', default=None)
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
 
    # validate paths 
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
        f_out = JSONStreamer(output_path)
        if not isfile(f_out.filename):
            raise FileNotFoundError(f"Output path not found: {args.output}")
        if not access(f_out.filename, W_OK):
            raise PermissionError(f"Output path not writable: {args.output}")

        # write context to output
        write_context(f_out, output_path)

        # write metadata to output
        write_metadata(f_out, vars(args))

    # load graph(s)
    print(f"importing {len(args.input)} graph(s)...", end=" ")
    kg = KnowledgeGraph(rng)
    kg.parse(args.input)
    print("done")

    # compute clauses
    f = generate(rng = rng, kg = kg, depths = args.depth,
                 min_support = args.min_support,
                 p_explore = args.p_explore,
                 p_extend = args.p_extend,
                 mode = args.mode,
                 max_length = args.max_size,
                 max_width = args.max_width,
                 multimodal = args.multimodal)

    if args.dry_run:
        exit(0)

    namespaces = {ns:pf for pf, ns in args.namespace}
    prefix_map = mkprefixes(kg.namespaces,
                            namespaces)

    # TODO: remove
    for i,c in enumerate(f.get(), 1):
        write_query(f_out, c, f"query_{i}", prefix_map)

    f_out.close()

