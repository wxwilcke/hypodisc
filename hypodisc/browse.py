#!/usr/bin/env python

import argparse
import codecs
import logging
from os import access, getcwd, R_OK
from os.path import isfile
import sqlite3
from multiprocessing import Process
from tempfile import NamedTemporaryFile
from threading import Timer
from typing import cast
import webbrowser

from flask import current_app, Flask, render_template, request

from rdf.formats import NTriples
from rdf.namespaces import RDF, XSD
from rdf.terms import IRIRef, Literal
from werkzeug.datastructures.structures import ImmutableMultiDict


REPO_IRI = IRIRef("https://gitlab.com/wxwilcke/hypodisc")
MLS_IRI = IRIRef("http://www.w3.org/ns/mls#")
DCT_IRI = IRIRef("http://purl.org/dc/terms/")

DB_PRIMARY_TABLE_NAME = "pattern"
DB_PRIMARY_TABLE_COLUMNS = ["id", "hasSupport", "hasLength", "hasWidth",
                            "hasDepth", "hasPattern", "hasDOTRepresentation"]
DB_PRIMARY_TABLE = """CREATE TABLE IF NOT EXISTS {}(
                                   {} INTEGER PRIMARY KEY,
                                   {} INTEGER NOT NULL,
                                   {} INTEGER NOT NULL,
                                   {} INTEGER NOT NULL,
                                   {} INTEGER NOT NULL,
                                   {} TEXT NOT NULL,
                                   {} TEXT NOT NULL
                                   )""".format(DB_PRIMARY_TABLE_NAME,
                                               *DB_PRIMARY_TABLE_COLUMNS)
DB_SEARCH_TABLE_NAME = "pattern_fts"
DB_SEARCH_TABLE_COLUMNS = ["id", "hasPattern"]
DB_SEARCH_TABLE = """CREATE VIRTUAL TABLE {} USING FTS5({}, {})""".format(
        DB_SEARCH_TABLE_NAME,
        *DB_SEARCH_TABLE_COLUMNS)

DB_QUERY_ALL = f"SELECT * FROM {DB_PRIMARY_TABLE_NAME} ORDER BY id ASC"
NUM_TRIPLES_PER_QUERY = len(DB_PRIMARY_TABLE_COLUMNS)

app = Flask(__name__, root_path=f'{getcwd()}/hypodisc/www/')
app.config['SERVER_NAME'] = '127.0.0.1:5000'


@app.route('/shutdown', methods=['GET'])
def shutdown():
    request.shutdown()
    

@app.route('/viewer', methods=['GET', 'POST'])
def viewer():
    offset = 0
    pagenum = current_app.pagenum  # type: ignore
    pagesize = current_app.pagesize  # type: ignore
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            if "id" in data.keys() and "selected" in data.keys():
                # save query checkbox ticked
                q_id = int(data["id"])
                q_tagged = data["selected"]
                if q_tagged is False:
                    current_app.favorites.remove(q_id)  # type: ignore
                else:
                    current_app.favorites.add(q_id)  # type: ignore
        else:
            if "reset" in request.form.keys():
                current_app.filters = current_app.defaults  # type: ignore
            elif "apply" in request.form.keys():
                current_app.filters = request.form  # type: ignore
            elif "clear_text_search" in request.form.keys():
                current_app.filters\
                        = {k: v for
                           k, v in current_app.filters.items()}  # type: ignore
                current_app.filters["text_search"] = ''  # type: ignore

            offset = 0  # reset when applying filters
    elif request.method == 'GET':
        if "page" in request.args.keys()\
                and int(request.args["page"]) != pagenum:
            page_diff = int(request.args["page"]) - pagenum

            pagenum += page_diff
            offset += page_diff * pagesize

        if "pagesize" in request.args.keys()\
                and int(request.args["pagesize"]) != pagesize:
            pagesize = int(request.args["pagesize"])
            current_app.pagesize = pagesize  # type: ignore

            pagenum = 1
            offset = 0  # reset when changing pagesize

    # generate query
    query = generate_query(current_app.filters, pagesize,  # type: ignore
                           offset, DB_PRIMARY_TABLE_NAME)

    # retrieve data
    data = current_app.cur.execute(query).fetchall()  # type: ignore

    # disable buttons if on first/last page
    first_page = False if offset > 0 else True
    exhausted = False if len(data) > 0 else True

    # disable save button if already saved
    saved = True
    unsaved_entries = True

    return render_template("default.html",
                           filename=current_app.filename,  # type: ignore
                           metadata=current_app.metadata,  # type: ignore
                           data=data,
                           offset=offset,
                           pagesize=pagesize,  # type: ignore
                           pagenum=pagenum,
                           max=max,
                           first_page=first_page,
                           exhausted=exhausted,
                           saved=saved,
                           unsaved_entries=unsaved_entries,
                           defaults=current_app.defaults,  # type: ignore
                           filters=current_app.filters,  # type: ignore
                           favorites=current_app.favorites)  # type: ignore


def generate_query(values: dict | ImmutableMultiDict, limit: int,
                   offset: int, table_name: str)\
                           -> str:
    """ Generate query based on provided values.

    :param values:
    :type values: ImmutableMultiDict
    :rtype: str
    """
    constraints = list()

    # full text search on FTS table
    search_query = values.get('text_search')
    if isinstance(search_query, str) and len(search_query) > 0:
        subquery = "SELECT id FROM {} WHERE hasPattern MATCH {}".format(
                    DB_SEARCH_TABLE_NAME, repr(search_query))

        constraints.append(f"id IN ({subquery})")

    # numerical filter settings
    for param in ["support", "length", "depth", "width"]:
        k_min, k_max = param + "_min", param + "_max"
        col_name = "has" + param.title()

        constraint = "{} BETWEEN {} AND {}".format(col_name,
                                                   values.get(k_min),
                                                   values.get(k_max))
        constraints.append(constraint)

    # order of results
    order = values.get("order_by")
    if order == "random":
        order = "RANDOM()"
    else:
        order_direction = values.get("order_by_dir")
        order += f" {order_direction}"  # type: ignore

    return """SELECT * FROM {} WHERE {} ORDER BY {} LIMIT {} OFFSET {}""".format(
            table_name,
            """ AND """.join(constraints),
            order, limit, offset)


def open_browser(port: int) -> None:
    """ Open localhost in default browser

    :param port:
    :type port: int
    :rtype: None
    """
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}/viewer')


def process_pattern(data: dict[str, Literal | int])\
        -> dict[str, str | int]:
    """ Process pattern by casting/transcoding the attribute values.

    :param data:
    :type data: dict[str, Literal | int]
    :rtype: dict[str, str | int]
    """
    out = dict()
    for attr, obj in data.items():
        if attr == 'id':
            out[attr] = obj

            continue

        if not isinstance(obj, Literal):
            continue

        value = obj.value
        if obj.datatype == XSD + 'nonNegativeInteger':
            value = int(value)
        if attr == 'hasPattern' and type(value) is str:
            value = codecs.decode(value, 'unicode_escape')
            value = '\n' + value
        if attr == 'hasDOTRepresentation' and type(value) is str:
            value = codecs.decode(value, 'unicode_escape')

        out[attr] = value

    return out


def write_table(conn: sqlite3.Connection, command: str) -> None:
    """ Write table to database.

    :param con:
    :type con: sqlite3.Connection
    :rtype: None
    """
    cur = conn.cursor()
    cur.execute(command)

    conn.commit()


def process_metadata(data: list[tuple[IRIRef, IRIRef, IRIRef | Literal]])\
        -> dict[str, list[str]]:
    """ Process metadata graph and return direct attribute to value map.

    :param data:
    :type data: list[tuple[IRIRef, IRIRef, IRIRef | Literal]]
    :rtype: dict[str, list[str]]
    """
    hyperparameters = list()
    base_ns, run_id = IRIRef("file://"), "UNKNOWN"
    for s, p, o in data:
        # extract base namespace and run identifier
        if p == RDF + "type" and o == MLS_IRI + "Run":
            base_ns = s
            _, run_id = split_IRI(s)

            continue

        # extract hyperparameter name
        if s == REPO_IRI and p == MLS_IRI + "hasHyperParameter"\
                and isinstance(o, IRIRef):
            _, hyperparameter = split_IRI(o)
            hyperparameters.append(hyperparameter)

    metadata = {'id': [run_id]}  # type: dict[str, list[str]]
    hyperparameters_IRIs = [base_ns + hp for hp in hyperparameters]
    for s, p, o in data:
        if s == base_ns:
            if o in hyperparameters_IRIs:
                # skip link to hyperparameters
                continue

            # add attribute name and values
            _, p_term = split_IRI(p)
            if p_term not in metadata.keys():
                metadata[p_term] = list()

            metadata[p_term].append(o.value)
        elif p == MLS_IRI + "hasValue":
            for i, hp_iri in enumerate(hyperparameters_IRIs):
                if s == hp_iri:
                    # directly add hyperparameter values
                    s_term = hyperparameters[i]
                    if s_term not in metadata.keys():
                        metadata[s_term] = list()

                    metadata[s_term].append(o.value)

                    break

    return metadata


def split_IRI(iri: IRIRef) -> tuple[str, str]:
    """ Split IRI in namespace and identity part.

    :param iri:
    :type iri: IRIRef
    :rtype: tuple[str, str]
    """
    iri_str = str(iri)
    try:
        ns, ident = iri_str.split('#')
        ns += '#'
    except ValueError:
        ident = iri_str.split('/')[-1]
        ns = iri_str[:-len(ident)]

    return ns, ident


def insert_data(conn: sqlite3.Connection, pattern_attrs: dict[str, str | int]):
    """ Insert pattern information into database.

    :param con:
    :type con: sqlite3.Connection
    :param pattern_attrs:
    :type pattern_attrs: dict[str, str]
    """
    cur = conn.cursor()

    values = list()
    for attr in DB_PRIMARY_TABLE_COLUMNS:
        values.append(pattern_attrs[attr])

    # insert row into table and commit
    pattern = '(' + '?, ' * (len(DB_PRIMARY_TABLE_COLUMNS) - 1) + '?)'
    cur.execute(f"INSERT INTO {DB_PRIMARY_TABLE_NAME} VALUES {pattern}",
                values)

    conn.commit()


def copy_db_columns(conn: sqlite3.Connection, table_destination: str,
                    table_source: str, column_names: list[str]) -> None:
    """ Populate virtual table for full text search.

    :param conn:
    :type conn: sqlite3.Connection
    :param table_name:
    :type table_name: str
    :param column_names:
    :type column_names: list[str]
    :rtype: None
    """
    cur = conn.cursor()
    cur.execute("INSERT INTO {} SELECT {} FROM {};".format(
        table_destination,
        ', '.join(column_names),
        table_source))

    conn.commit()


def update_defaults(defaults: dict[str, int], pattern: dict[str, str | int])\
        -> None:
    """ Update default min/max values.

    :param defaults:
    :type defaults: dict[str, int]
    :param pattern:
    :type pattern: dict[str, str | int]
    :rtype: None
    """
    support = cast(int, pattern["hasSupport"])
    if support < defaults["support_min"] or defaults["support_min"] < 0:
        defaults["support_min"] = support
    if support > defaults["support_max"]:
        defaults["support_max"] = support

    length = cast(int, pattern["hasLength"])
    if length < defaults["length_min"] or defaults["length_min"] < 0:
        defaults["length_min"] = length
    if length > defaults["length_max"]:
        defaults["length_max"] = length

    depth = cast(int, pattern["hasDepth"])
    if depth < defaults["depth_min"] or defaults["depth_min"] < 0:
        defaults["depth_min"] = depth
    if depth > defaults["depth_max"]:
        defaults["depth_max"] = depth

    width = cast(int, pattern["hasWidth"])
    if width < defaults["width_min"] or defaults["width_min"] < 0:
        defaults["width_min"] = width
    if width > defaults["width_max"]:
        defaults["width_max"] = width


def populate_db(conn: sqlite3.Connection, g: NTriples,
                defaults: dict[str, int])\
        -> list[tuple[IRIRef, IRIRef, IRIRef | Literal]]:
    """ Iterate over the graph and add all patterns to the database. Keep track
    of a pattern's attributes to support unsorted entries. Returns the metadata

    :param con:
    :type con: sqlite3.Connection
    :param g:
    :type g: NTriples
    :rtype: list[tuple[str, str, str]]
    """
    patterns = dict()
    metadata = list()
    for s, p, o in g.parse():
        _, ident = split_IRI(s)
        if not ident.lower().startswith("query"):
            # everything that is not a query is metadata
            metadata.append((s, p, o))

            continue

        if ident not in patterns.keys():
            _, ident_num = ident.split('_')
            patterns[ident] = {'id': int(ident_num)}

        # add attribute and its value
        _, attr = split_IRI(p)
        patterns[ident][attr] = o

        if len(patterns[ident]) >= NUM_TRIPLES_PER_QUERY:
            pattern = process_pattern(patterns[ident])
            update_defaults(defaults, pattern)
            insert_data(conn, pattern)

            del patterns[ident]  # no longer needed so free memory

    return metadata


def setup_logger(verbose: bool) -> None:
    """ Setup logger

    :param verbose: print debug messages
    :type verbose: bool
    :rtype: None
    """
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)


def create_app(filename: str, conn: sqlite3.Connection,
               metadata: dict[str, list[str]], favorites: set[int],
               defaults: dict[str, int], pagesize: int) -> Flask:
    """ Set up app variables.

    :param filename:
    :type filename: str
    :param conn:
    :type conn: sqlite3.Connection
    :param metadata:
    :type metadata: dict[str, list[str]]
    :rtype: Flask
    """
    with app.app_context():
        current_app.filename = filename  # type: ignore
        current_app.cur = conn.cursor()  # type: ignore
        current_app.metadata = metadata  # type: ignore
        current_app.pagenum = 1  # type: ignore
        current_app.pagesize = pagesize  # type: ignore
        current_app.defaults = defaults  # type: ignore
        current_app.filters = defaults  # type: ignore
        current_app.favorites = favorites  # type: ignore

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HypoDisc")
    parser.add_argument("input", help="An N-Triple file containing queries "
                        + "with hypotheses", type=str)
    parser.add_argument("--base_ns", help="Provide a custom base namespace",
                        default=None)
    parser.add_argument("--pagesize", help="Number of queries to show per "
                        + "page.", default=25, type=int)
    parser.add_argument("--port", help="Change the default port",
                        default=5000, type=int)
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

    with NamedTemporaryFile(suffix='.db') as f_temp:
        # initiate temporary database
        conn = sqlite3.connect(f_temp.name, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # return rows as dicts

        write_table(conn, DB_PRIMARY_TABLE)  # set up primary table
        write_table(conn, DB_SEARCH_TABLE)  # set up FTS table

        # default min/max values
        defaults = {"support_min": -1, "support_max": -1,
                    "length_min": -1, "length_max": -1,
                    "depth_min": -1, "depth_max": -1,
                    "width_min": -1, "width_max": -1,
                    "order_by": "id", "order_by_dir": "ASC"}

        metadata = dict()
        with NTriples(args.input) as g:
            # populate database with patterns
            metadata = populate_db(conn, g, defaults)

        # copy patterns and IDs to FTS table
        copy_db_columns(conn, DB_SEARCH_TABLE_NAME,
                        DB_PRIMARY_TABLE_NAME, ['id', 'hasPattern'])

        # extract relevant metadata
        metadata = process_metadata(metadata)

        # store IDs of queries to save later
        favorites = set()  # type: set[int]

        # open viewer in default browser
        if not args.suppress_browser:
            pass
            Timer(1, open_browser, [args.port]).start()

        # run flask app
        web_app = create_app(args.input, conn, metadata, favorites,
                             defaults, args.pagesize)

        with app.app_context():
            web_app.run(debug=args.verbose, port=args.port)
