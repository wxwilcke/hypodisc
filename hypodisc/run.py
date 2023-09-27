#!/usr/bin/env python

import argparse
import logging
from os import access, getcwd, R_OK, W_OK
from os.path import isdir, isfile, splitext
from sys import maxsize, exit
from time import time

from core.sequential import generate
#from ui import _LEFTARROW, _PHI, generate_label_map, pretty_clause
from core.utils import (floatProbabilityArg, integerRangeArg, read_version,
                        rng_set_seed)
from data.graph import KnowledgeGraph
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
    parser.add_argument("-s", "--min_support", help="Minimal clause support.",
                        type=int, required=True)
    parser.add_argument("-c", "--min_confidence", help="Minimal clause"
                        + " confidence", type=int, required=True)
    parser.add_argument("-o", "--output", help="Path to write output to.",
                        type=str, default=getcwd())
    parser.add_argument("input", help="One or more knowledge graphs in"
                        + "(gzipped) NTriple or NQuad serialization format.",
                        nargs='+')
    parser.add_argument("--max_size", help="Maximum context size",
                        type=int, required=False, default=maxsize)
    parser.add_argument("--max_width", help="Maximum width of shell",
                        type=int, required=False, default=maxsize)
    parser.add_argument("--mode", help="A[box], T[box], or B[oth] as"
                        + " candidates for head and body", choices = ["AA",
                        "AT", "TA", "TT", "AB", "BA", "TB", "BT", "BB"],
                        type=str, default="BB")
    parser.add_argument("--multimodal", help="Enable multimodal support",
                        required=False, action='store_true')
    parser.add_argument("--p_explore", help="Probability of exploring "
                        + "candidate endpoint.", type=floatProbabilityArg,
                        required=False, default=1.0)
    parser.add_argument("--p_extend", help="Probability of extending at "
                        + "candidate endpoint.", type=floatProbabilityArg,
                        required=False, default=1.0)
    parser.add_argument("--prune", help="Whether to prune the result."
                        + " Defaults to true.", type=bool, required=False,
                        default=True, action=argparse.BooleanOptionalAction)
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

    # validate paths 
    if isdir(args.output):
        # create new file with default name
        args.output = mkfile(args.output, OUTPUT_NAME, OUTPUT_EXT)
    if not args.output.endswith(OUTPUT_EXT):
        _, ext = splitext(args.output)
        raise UnsupportedSerializationFormat(f"Specified output path "
                                            "has unexpected extension: {ext}")
    for filename in args.input:
        if not isfile(filename): 
            raise FileNotFoundError(f"Input path not found: {filename}")
        if not access(filename, R_OK):
            raise PermissionError(f"Input path not readable: {filename}")

    f_out = open(args.output, 'w')  # create file on disk
    if not isfile(args.output):
        raise FileNotFoundError(f"Output path not found: {args.output}")
    if not access(args.output, W_OK):
        raise PermissionError(f"Output path not writable: {args.output}")

    # load graph(s)
    print(f"importing {len(args.input)} graph(s)...", end=" ")
    kg = KnowledgeGraph(rng)
    kg.parse(args.input)
    print("done")

    # compute clauses
    f = generate(rng = rng, kg = kg, depths = args.depth,
                 min_support = args.min_support,
                 min_confidence = args.min_confidence,
                 p_explore = args.p_explore,
                 p_extend = args.p_extend,
                 prune = args.prune, mode = args.mode,
                 max_length = args.max_size,
                 max_width = args.max_width,
                 multimodal = args.multimodal)

    if args.dry_run:
        exit(0)

    # TODO: remove
    for c in f.get():
        print(c)

#    print("storing results...", end=" ")
#    # store clauses
#    if args.output == "pkl":
#        pickle.dump(f, open("./generation_forest(d{}s{}c{})_{}.pkl".format(str(args.depth)[5:],
#                                                                           str(args.min_support),
#                                                                           str(args.min_confidence),
#                                                                           timestamp), "wb"))
#    else:
#        ns_dict = {v:k for k,v in g.namespaces()}
#        label_dict = generate_label_map(g)
#        with open("./generation_forest(d{}s{}c{})_{}.tsv".format(str(args.depth)[5:],
#                                                                 str(args.min_support),
#                                                                 str(args.min_confidence),
#                                                                 timestamp), "w") as ofile:
#            writer = csv.writer(ofile, delimiter="\t")
#            writer.writerow(['Depth', 'P_domain', 'P_range', 'Supp', 'Conf', 'Head', 'Body'])
#            for c in f.get():
#                depth = max(c.body.distances.keys())
#                bare = pretty_clause(c, ns_dict, label_dict).split("\n"+_PHI+": ")[-1].split(" "+_LEFTARROW+" ")
#                writer.writerow([depth,
#                                 c.domain_probability, c.range_probability,
#                                 c.support, c.confidence,
#                                 bare[0], bare[1]])
#
#    print("done")
