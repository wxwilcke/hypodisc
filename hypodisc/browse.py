#!/usr/bin/env python

import argparse
import codecs
import logging
from os import access, getcwd, R_OK
from os.path import isfile
import sqlite3
from tempfile import NamedTemporaryFile
from threading import Timer
import webbrowser

from flask import current_app, Flask, render_template, url_for, request

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
DB_SEARCH_TABLE = """CREATE VIRTUAL TABLE IF NOT EXISTS {} USING FTS5(
                                   {} INTEGER PRIMARY KEY,
                                   {} TEXT NOT NULL
                                   )""".format(DB_SEARCH_TABLE_NAME,
                                               *DB_SEARCH_TABLE_COLUMNS)

DB_QUERY_ALL = f"SELECT * FROM {DB_PRIMARY_TABLE_NAME}"
NUM_TRIPLES_PER_QUERY = len(DB_PRIMARY_TABLE_COLUMNS)

app = Flask(__name__, root_path=f'{getcwd()}/hypodisc/www/')
app.config['SERVER_NAME'] = '127.0.0.1:5000'


@app.route('/', methods=['GET', 'POST'])
def viewer():
    query = DB_QUERY_ALL
    if request.method == 'POST':
        filter_config = request.form
        query = generate_query(filter_config)

    # retrieve data
    qcur = current_app.cur.execute(query)  # type: ignore

    return render_template("default.html",
                           filename=current_app.filename,  # type: ignore
                           metadata=current_app.metadata,  # type: ignore
                           cursor=qcur,
                           pagesize=current_app.pagesize)  # type: ignore


def generate_query(config: ImmutableMultiDict) -> str:
    search_query = config.get('text_search')
    if isinstance(search_query, str) and len(search_query) > 0:
        pass


def open_browser(port: int) -> None:
    """ Open localhost in default browser

    :param port:
    :type port: int
    :rtype: None
    """
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}')


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


def populate_db(conn: sqlite3.Connection, g: NTriples)\
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
               metadata: dict[str, list[str]], pagesize: int) -> Flask:
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
        current_app.pagesize = pagesize  # type: ignore

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

        metadata = dict()
        with NTriples(args.input) as g:
            # populate database with patterns
            metadata = populate_db(conn, g)

        # copy patterns and IDs to FTS table
        copy_db_columns(conn, DB_SEARCH_TABLE_NAME,
                        DB_PRIMARY_TABLE_NAME, ['id', 'hasPattern'])

        # extract relevant metadata
        metadata = process_metadata(metadata)

        # open viewer in default browser
        if not args.suppress_browser:
            pass
            Timer(1, open_browser, [args.port]).start()

        # run flask app
        web_app = create_app(args.input, conn, metadata, args.pagesize)
        with app.app_context():
            web_app.run(debug=args.verbose, port=args.port)
