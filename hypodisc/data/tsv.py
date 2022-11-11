#!/usr/bin/python3

import csv
from tempfile import TemporaryFile


class TSV:
    """ TSV Class
    Create files on the fly with tab-separated values
    """

    _tsv = None
    _writer = None

    def __init__(self, path=None, mode='r', temporary=False):
        if temporary:
            if 'r' in mode:
                raise ValueError("::Cannot read from temporary file")
            self._tsv = TemporaryFile(mode='w')
            self._writer = csv.writer(self._tsv, delimiter="\t")

            return

        if path is None:
            raise ValueError("::No path supplied")

        self._tsv = open(path, mode)

        if 'r' in mode:
            return self.read()
        else:
            self._writer = csv.writer(self._tsv, delimiter="\t")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._tsv.close()

    def read(self):
        return csv.reader(self._tsv, delimiter="\t")

    def writerow(self, row):
        self._writer.writerow(row)
        self._tsv.flush()

    def writerows(self, rows):
        self._writer.writerows(rows)
        self._tsv.flush()
