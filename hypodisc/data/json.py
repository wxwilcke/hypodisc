#!/usr/bin/env python

from __future__ import annotations
from datetime import datetime
from enum import Enum
from getpass import getuser
from numbers import Number
from io import TextIOWrapper
from pathlib import Path
from os.path import basename
from typing import Any, Self, Union

from hypodisc.core.structures import GraphPattern


CONTEXT = { "@version": 1.1,
            "dct": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

METADATA = { "@id": "",
             "@type": "dct:Dataset",
             "dct:format": { 
                    "@id": ("https://www.iana.org/assignments/media-types/"
                            "application/ld+json"),
                    "@type": "dct:MediaType" }, 
             "dct:description": ("A collection of SPARQL queries generated "
                                 "automatically by HypoDisc: a tool, funded "
                                 "by CLARIAH and developed by the DataLegend "
                                 "team (Xander Wilcke, Richard Zijdeman, Rick "
                                 "Mourits, Auke Rijpma, and Sytze van Herck), "
                                 "that can discover novel and "
                                 "potentially-interesting graph patterns "
                                 "in multimodal knowledge graphs which can be "
                                 "used by experts and scholars to form new "
                                 "research hypotheses, to support existing "
                                 "ones, or to gain insight into their data."),
             "rdfs:label": "A collection of SPARQL queries",
             "prov:wasGeneratedBy": {"@id": "base:HypothesisDiscovery"},
             "base:HypothesisDiscovery": {
                 "@type": "prov:Activity",
                 "prov:used": { "@id": "https://gitlab.com/wxwilcke/hypodisc",
                                "@type": "dct:Software" },
                 "prov:wasStartedBy": f"{getuser().title()}",
                 "prov:startedAtTime": f"{datetime.isoformat(datetime.now())}"
                 }
            }

def write_context(j:JSONStreamer, path:Path, update:dict = {}) -> None:
    context = { k:v for k,v in CONTEXT.items() }
    context.update(update)

    context["@base"] = path.absolute().as_uri()
    context["base"] = path.absolute().as_uri()

    j.write_dict("@context", context)

def write_metadata(j:JSONStreamer, parameters:dict[str,Any],\
        update:dict = {}) -> None:
    metadata = { k:v for k,v in METADATA.items() }
    metadata.update(update)

    source = [basename(path) for path in parameters['input']]
    metadata['base:HypothesisDiscovery'].update({
        "prov:hadPrimarySource": source if len(source) > 1 else source.pop() })

    for k,v in metadata.items():
        j.write_key_value(k, v)

def write_query(j:JSONStreamer, pattern:GraphPattern, name:str) -> None:
    query = { "support": pattern.support,
              "length": len(pattern),
              "width": pattern.width(),
              "depth": pattern.depth(),
              "query": pattern.as_query() }

    j.write_dict(f"@base:{name}", query)

class JSONStreamer():
    class DataStructures(Enum):
        LIST = 0
        DICT = 1


    def __init__(self, path:Union[str, Path], indent_spaces:int = 2) -> None:
        """ A Lightweight JSON Streamer

        :param path:
        :type path: Union[str, Path]
        :param indent_spaces:
        :type indent_spaces: int
        :rtype: None
        """
        self._indent_level = list()
        self._indent_count = list()
        self._indent_spaces = indent_spaces

        path = Path(path)

        self.open(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback)\
            -> None:
        self.close()

    def open(self, path:Path) -> None:
        """ Open new JSON file

        :param path:
        :type path: Path
        :rtype: None
        """
        self.filename = path.absolute()
        self._file = TextIOWrapper(open(path, 'wb'),
                                   line_buffering = True)
        self._indent_in(JSONStreamer.DataStructures.DICT)

    def close(self) -> None:
        """ Close current JSON file

        :rtype: None
        """
        for _ in reversed(self._indent_level):
            self._indent_out()

        self._file.close()

    def _indent_in(self, struc:JSONStreamer.DataStructures,
                  inline:bool = False) -> None:
        """ Insert new indentation level and move to that level.

        :param struc:
        :type struc: JSONStreamer.DataStructures
        :param inline:
        :type inline: bool
        :rtype: None
        """
        if struc == JSONStreamer.DataStructures.LIST:
            char = '['
        elif struc == JSONStreamer.DataStructures.DICT:
            char = '{'
        else:
            raise Exception()

        num_spaces = 0
        if not inline:
            num_spaces = len(self._indent_level) * self._indent_spaces

        self._file.write(' ' * num_spaces + f"{char}")
        self._indent_level.append(struc)
        self._indent_count.append(0)
        
    def _indent_out(self) -> None:
        """ Close current indentation level and return to previous one.

        :rtype: None
        """
        if len(self._indent_level) <= 0:
            raise Exception(f"Already at lowest indent level")

        del self._indent_count[-1]
        struc = self._indent_level.pop()
        num_spaces = len(self._indent_level) * self._indent_spaces
        if struc == JSONStreamer.DataStructures.LIST:
            char = ']'
        elif struc == JSONStreamer.DataStructures.DICT:
            char = '}'
        else:
            raise Exception()

        self._file.write('\n' + ' ' * num_spaces + f"{char}")

    def write_list(self, key:str, values:Union[list, set, tuple]) -> None:
        """ Write a list to the output file.

        :param key:
        :type key: str
        :param values:
        :type values: Union[list, set, tuple]
        :rtype: None
        """
        if len(self._indent_level) == 1 and self._indent_count[-1] <= 0:
            pre = '\n'
        else:
            pre = ",\n"

        self._write_list(key, values, pre)

    def write_dict(self, key:str, values:dict) -> None:
        """ Write a dictionary to the output file.

        :param key:
        :type key: str
        :param values:
        :type values: dict
        :rtype: None
        """
        if len(self._indent_level) == 1 and self._indent_count[-1] <= 0:
            pre = '\n'
        else:
            pre = ",\n"

        self._write_dict(key, values, pre)

    def _write_list(self, key:Union[str,None],
                    values:Union[list, set, tuple],
                    pre:str = '\n', inline:bool = False) -> None:
        """ Write an inline or regular list to output.

        :param key:
        :type key: Union[str,None]
        :param values:
        :type values: Union[list, set, tuple]
        :param pre:
        :type pre: str
        :param inline:
        :type inline: bool
        :rtype: None
        """
        num_spaces = len(self._indent_level) * self._indent_spaces
        if key is not None:
            self._file.write(pre + ' ' * num_spaces + f"\"{key}\": ")
        elif not inline:
            self._file.write(pre + ' ' * num_spaces)

        self._indent_count[-1] += 1
        self._indent_in(JSONStreamer.DataStructures.LIST,
                        inline = True)
        num_spaces = len(self._indent_level) * self._indent_spaces
        for v in values:
            pre = ",\n" if self._indent_count[-1] > 0 else '\n'
            self._indent_count[-1] += 1

            if type(v) is dict:
                self._write_dict(None, v, pre)
            elif type(v) in [list, set, tuple]:
                self._write_list(None, v, pre)
            else:
                v = self.cast(v)
                self._file.write(pre + ' ' * num_spaces + f"{v}")

        self._indent_out()

    def _write_dict(self, key:Union[str, None], values:dict,
                    pre:str = '\n', inline:bool = False) -> None:
        """ Write an inline or regular list to output.

        :param key:
        :type key: Union[str, None]
        :param values:
        :type values: dict
        :param pre:
        :type pre: str
        :param inline:
        :type inline: bool
        :rtype: None
        """
        num_spaces = len(self._indent_level) * self._indent_spaces
        if key is not None:
            self._file.write(pre + ' ' * num_spaces + f"\"{key}\": ")
        elif not inline:
            self._file.write(pre + ' ' * num_spaces)

        self._indent_count[-1] += 1
        self._indent_in(JSONStreamer.DataStructures.DICT,
                        inline = True)
        num_spaces = len(self._indent_level) * self._indent_spaces
        for k, v in values.items():
            pre = ",\n" if self._indent_count[-1] > 0 else '\n'
            self._indent_count[-1] += 1

            self._file.write(pre + ' ' * num_spaces + f"\"{k}\": ")
            if type(v) is dict:
                self._write_dict(None, v, inline = True)
            elif type(v) in [list, set, tuple]:
                self._write_list(None, v, inline = True)
            else:
                v = self.cast(v)
                self._file.write(f"{v}")

        self._indent_out()

    def write_key_value(self, key:str, value:Any) -> None:
        """ Write a key-value pair to output.

        :param key:
        :type key: str
        :param value:
        :type value: Any
        :rtype: None
        """
        pre = ",\n" if self._indent_count[-1] > 0 else '\n'
        self._indent_count[-1] += 1
        
        num_spaces = len(self._indent_level) * self._indent_spaces
        self._file.write(pre + ' ' * num_spaces + f"\"{key}\": ")
        if type(value) is dict:
            self._write_dict(None, value, inline = True)
        elif type(value) in [list, set, tuple]:
            self._write_list(None, value, inline = True)
        else:
            value = self.cast(value)
            self._file.write(f"{value}")

    def cast(self, value:Any) -> Union[str, Number]:
        """ Cast Python objects to JSON objects.

        :param value:
        :type value: Any
        :rtype: Union[str, Number]
        """
        if value is None:
            value = "null"
        elif type(value) is bool:
            value = str(bool).lower()
        elif not str(value).isdigit():
            value = f"\"{value}\""

        return value      
