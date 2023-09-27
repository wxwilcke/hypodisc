#!/usr/bin/env python

from __future__ import annotations
from enum import Enum
from numbers import Number
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Self, Union


INDENT_SPACES = 2

class JSONStreamer():
    class DataStructures(Enum):
        LIST = 0
        DICT = 1


    def __init__(self, path:Union[str, Path]) -> None:
        """ A Lightweight JSON Streamer

        :param path:
        :type path: Union[str, Path]
        :rtype: None
        """
        self._indent_level = list()
        self._indent_count = list()
        self.open(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback)\
            -> None:
        self.close()

    def open(self, path:Union[str, Path]) -> None:
        """ Open new JSON file

        :param path:
        :type path: Union[str, Path]
        :rtype: None
        """
        self._file = TextIOWrapper(open(path, 'wb'),
                                   line_buffering = True)
        self.indent_in(JSONStreamer.DataStructures.DICT)

    def close(self) -> None:
        """ Close current JSON file

        :rtype: None
        """
        for _ in reversed(self._indent_level):
            self.indent_out()

        self._file.close()

    def indent_in(self, struc:JSONStreamer.DataStructures,
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
            num_spaces = len(self._indent_level) * INDENT_SPACES

        self._file.write(' ' * num_spaces + f"{char}")
        self._indent_level.append(struc)
        self._indent_count.append(0)
        
    def indent_out(self) -> None:
        """ Close current indentation level and return to previous one.

        :rtype: None
        """
        if len(self._indent_level) <= 0:
            raise Exception(f"Already at lowest indent level")

        del self._indent_count[-1]
        struc = self._indent_level.pop()
        num_spaces = len(self._indent_level) * INDENT_SPACES
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
        num_spaces = len(self._indent_level) * INDENT_SPACES
        if key is not None:
            self._file.write(pre + ' ' * num_spaces + f"\"{key}\": ")
        elif not inline:
            self._file.write(pre + ' ' * num_spaces)

        self._indent_count[-1] += 1
        self.indent_in(JSONStreamer.DataStructures.LIST,
                       inline = True)
        num_spaces = len(self._indent_level) * INDENT_SPACES
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

        self.indent_out()

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
        num_spaces = len(self._indent_level) * INDENT_SPACES
        if key is not None:
            self._file.write(pre + ' ' * num_spaces + f"\"{key}\": ")
        elif not inline:
            self._file.write(pre + ' ' * num_spaces)

        self._indent_count[-1] += 1
        self.indent_in(JSONStreamer.DataStructures.DICT,
                       inline = True)
        num_spaces = len(self._indent_level) * INDENT_SPACES
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

        self.indent_out()

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
        
        num_spaces = len(self._indent_level) * INDENT_SPACES
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
