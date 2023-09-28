#! /usr/bin/env python

from pathlib import Path


def mkfile(directory:str, basename:str, extension:str) -> Path:
    """ Return path to a new file. Adds numerical suffix if
        the file already exists.

    :param directory:
    :type directory: str
    :param basename:
    :type basename: str
    :param extension:
    :type extension: str
    :rtype: Path
    """
    if not extension.startswith('.'):
        extension = '.' + extension

    out = Path(directory).joinpath(basename).with_suffix(extension)
    if not out.exists():
        return out

    suffix = 1
    while out.exists():
        outname = f"{basename}-{suffix}"
        out = Path(directory).joinpath(outname).with_suffix(extension)

        suffix += 1

    return out

class UnsupportedSerializationFormat(Exception):
    pass
