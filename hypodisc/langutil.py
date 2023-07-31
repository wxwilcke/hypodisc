#! /usr/bin/env python

from typing import Literal, Optional, Union


REGEX_ILLEGAL_CHARS = ['[',']','\\','^','*','+','?','{',
                       '}','|','(',')','$','.', '"']
REGEX_WHITE_SPACE = "\s"
REGEX_LOWER = "[a-z]"
REGEX_UPPER = "[A-Z]"
REGEX_DIGIT = "[0-9]"
REGEX_PUNCT = "[(\.|\?|!)]"  # subset
REGEX_OTHER = "[^a-zA-Z0-9\.\?! ]"


def pattern_sequence(pattern:str) -> list:
    seq = pattern[1, -1].split(REGEX_WHITE_SPACE)
    for i, subpattern in enumerate(seq):
        char_class = ""

        j = 0
        while j < len(subpattern):
            c = subpattern[j]
            if c == '{' and (j > 0 and subpattern[j-1] != '\\'):
                # quantifier begin
                k = 0
                freq = 0
                while k < len(subpattern):
                    q = subpattern[k]
                    if q == '}':
                        # quantifier ends
                        break

                    freq += int(q)  # update quantifier
                    k += 1

                j += k + 1
            j += 1

def generalize_regex(patterns):
    generalized_patterns = set()

    subpattern_list = list()
    for pattern in patterns:
        if len(pattern) <= 2:
            # empty string
            continue

        subpatterns = pattern[1:-1].split('\s')
        if subpatterns[-1][:-3].endswith('[(\.|\?|!)]'):
            end = subpatterns[-1][-14:]
            subpatterns[-1] = subpatterns[-1][:-14]
            subpatterns.append(end)

        for i, subpattern in enumerate(subpatterns):
            if len(subpattern_list) <= i:
                subpattern_list.append(dict())

            char_pattern = subpattern[:-3]
            if char_pattern not in subpattern_list[i].keys():
                subpattern_list[i][char_pattern] = list()
            subpattern_list[i][char_pattern].append(int(subpattern[-2:-1]))

    subpattern_cluster_list = list()
    for i, subpatterns in enumerate(subpattern_list):
        if len(subpattern_cluster_list) <= i:
            subpattern_cluster_list.append(dict())

        for subpattern, lengths in subpatterns.items():
            if subpattern not in subpattern_cluster_list[i].keys():
                subpattern_cluster_list[i][subpattern] = list()

            if len(lengths) <= 2 or len(set(lengths)) == 1:
                clusters = [(min(lengths), max(lengths))]
            else:
                clusters = [(int(a), int(b)) for a,b in
                            numeric_clusters(np.array(lengths), acc=0)]

            subpattern_cluster_list[i][subpattern] = clusters

    for pattern in patterns:
        subpatterns = pattern[1:-1].split('\s')
        if subpatterns[-1][:-3].endswith('[(\.|\?|!)]'):
            end = subpatterns[-1][-14:]
            subpatterns[-1] = subpatterns[-1][:-14]
            subpatterns.append(end)
        generalized_patterns |= combine_regex(subpatterns,
                                              subpattern_cluster_list)

    return generalized_patterns

def combine_regex(subpatterns, subpattern_cluster_list, _pattern='', _i=0):
    if len(subpatterns) <= 0:
        return {_pattern+'$'}

    patterns = set()
    char_pattern = subpatterns[0][:-3]
    if char_pattern in subpattern_cluster_list[_i].keys():
        for a,b in subpattern_cluster_list[_i][char_pattern]:
            if a == b:
                length = '{' + str(a) + '}'
            else:
                length = '{' + str(a) + ',' + str(b) + '}'

            if _i <= 0:
                pattern = '^' + char_pattern + length
            elif char_pattern == "[(\.|\?|!)]":
                pattern = _pattern + char_pattern + length
            else:
                pattern = _pattern + '\s' + char_pattern + length

            patterns |= combine_regex(subpatterns[1:], subpattern_cluster_list,
                                      pattern, _i+1)

    return patterns


def generate_regex(s:str) -> RegexPattern:
    """ Generate regular expresion that fits string. 

    :param s:
    :type s: str
    :rtype: RegexPattern
    """
    # remove unecessary white space
    s = ' '.join(s.split())
    slen = len(s)

    pattern = RegexPattern()
    if slen <= 0:
        # empty string
        return pattern

    char_set = mkcharset(char_to_regex(s[0]))
    char_set.add(s[0])
    for i in range(1, slen):
        symbol_char_set = char_to_regex(s[i])
        if symbol_char_set == char_set.value:
            # pattern continues still
            char_set.add(s[i])
        
            # account for final character
            if i >= slen - 1:
                pattern.add(char_set)

            continue

        # add interupted pattern to output
        pattern.add(char_set)

        # start next char set
        char_set = mkcharset(symbol_char_set)
        char_set.add(s[i])

        # account for final character
        if i >= slen - 1:
            pattern.add(char_set)

    return pattern

def mkcharset(char_set:str) -> Union[RegexChar, RegexCharSet,
                                     RegexCharRangedSet]:
    """mkcharset.

    :param char_set:
    :type char_set: str
    :rtype: Union[RegexChar, RegexCharSet,
                                         RegexCharRangedSet]
    """
    if char_set in {REGEX_DIGIT, REGEX_LOWER, REGEX_UPPER}:
        char_set_object = RegexCharRangedSet(char_set)
    elif char_set == REGEX_PUNCT:
        char_set_object = RegexCharSet(char_set)
    else:  # white space and other non-word chars
        char_set_object = RegexChar(char_set)

    return char_set_object

def char_to_regex(c:str) -> str:
    """ Return suitable character class or (escaped) char if none fit.

    :param c:
    :type c: str
    :rtype: str
    """
    char_class = character_class(c)
    if char_class == REGEX_OTHER:
        if c in REGEX_ILLEGAL_CHARS:
            # escape char
            c = "\\" + c

        return c

    return char_class

def character_class(c:str) -> str:
    """ Infer character class.

    :param c:
    :type c: str
    :rtype: str
    """
    if c.isalpha():
        char_class = REGEX_LOWER if c.islower() else REGEX_UPPER
    elif c.isdigit():
        char_class = REGEX_DIGIT
    elif c.isspace():
        char_class = REGEX_WHITE_SPACE
    elif c == "." or c == "?" or c == "!":
        char_class = REGEX_PUNCT
    else:
        char_class = REGEX_OTHER

    return char_class

class RegexChar():
    def __init__(self, value:str) -> None:
        """__init__.

        :param value:
        :type value: str
        :rtype: None
        """
        self.value = value
        self.count = 0

    def add(self, char:Optional[str]) -> None:
        """ Add character to set

        :param char:
        :type char: str
        :rtype: None
        """
        self.count += 1

    def exact(self) -> str:
        """exact.

        :rtype: str
        """
        return self.value + '{' + str(self.count) + '}'

    def equiv(self, other:Union['RegexChar', 'RegexCharSet',
                                'RegexCharRangedSet']) -> bool:
        """equiv.

        :param other:
        :type other: Union['RegexChar', 'RegexCharSet',
                                        'RegexCharRangedSet']
        :rtype: bool
        """
        return self.value == other.value

    def __eq__(self, other:Union['RegexChar', 'RegexCharSet',
                                 'RegexCharRangedSet']) -> bool:
        """__eq__.

        :param other:
        :type other: Union['RegexChar', 'RegexCharSet',
                                         'RegexCharRangedSet']
        :rtype: bool
        """
        return self.exact() == other.exact()

    def __repr__(self) -> str:
        """__repr__.

        :rtype: str
        """
        return str(self)

    def __str__(self) -> str:
        """ Print character class.

        :rtype: str
        """
        out = self.value
        if self.count > 1:
            out +=  '{' + str(self.count) + '}'

        return out


    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return self.count

class RegexCharSet():
    def __init__(self, value:str) -> None:
        """__init__.

        :param value:
        :type value: str
        :rtype: None
        """
        self.value = value
        self.count = 0

    def add(self, char:Optional[str]) -> None:
        """ Add character to set

        :param char:
        :type char: str
        :rtype: None
        """
        self.count += 1

    def exact(self) -> str:
        """exact.

        :rtype: str
        """
        return str(self)

    def equiv(self, other:Union['RegexChar', 'RegexCharSet',
                                'RegexCharRangedSet']) -> bool:
        """equiv.

        :param other:
        :type other: Union['RegexChar', 'RegexCharSet',
                                        'RegexCharRangedSet']
        :rtype: bool
        """
        return self.value == other.value

    def __eq__(self, other:Union['RegexChar', 'RegexCharSet',
                                 'RegexCharRangedSet']) -> bool:
        """__eq__.

        :param other:
        :type other: Union['RegexChar', 'RegexCharSet',
                                         'RegexCharRangedSet']
        :rtype: bool
        """
        return self.exact() == other.exact()

    def __repr__(self) -> str:
        """__repr__.

        :rtype: str
        """
        return str(self)

    def __str__(self) -> str:
        """ Print character class.

        :rtype: str
        """
        out = self.value
        if self.count > 1:
            out +=  '{' + str(self.count) + '}'

        return out

    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return self.count

class RegexCharRangedSet(RegexCharSet):
    def __init__(self, value:str) -> None:
        """ Regex character class.

        :param char_class:
        :type char_class: str
        :rtype: None
        """
        super().__init__(value)
        self.values = set()
        self.count = 0

    def add(self, char:str) -> None:
        """ Add character to set

        :param char:
        :type char: str
        :rtype: None
        """
        self.values.add(char)
        self.count += 1

    def merge(self, other:'RegexCharRangedSet') -> 'RegexCharRangedSet':
        """merge.

        :param other:
        :type other: 'RegexCharRangedSet'
        :rtype: 'RegexCharRangedSet'
        """
        char_class = self.value[:-1] + other.value[1:]

        merged = RegexCharRangedSet(char_class)
        merged.values = set.union(self.values, other.values)
        merged.count = self.count + other.count

        return merged

    def exact(self) -> str:
        """exact.

        :rtype: str
        """
        out = ''
        if self.value == "[0-9]":
            begin = min(self.values)
            end = max(self.values)

            if begin == end:
                out = f'{begin}'
            else:
                out = f'[{begin}-{end}]'

            if self.count > 1:
                out +=  '{' + str(self.count) + '}'
        else:
            sets = [(ord(self.value[i]), ord(self.value[i+2]))
                    for i in range(1, len(self.value) - 1, 3)]

            values = [ord(v) for v in self.values]
            for value_set in sets:
                members = [i for i in values if i in range(value_set[0],
                                                           value_set[1] + 1)]
                begin = chr(min(members))
                end = chr(max(members))

                if begin == end:
                    out = f'{begin}'
                else:
                    out += f'[{begin}-{end}]'

                if self.count > 1:
                    out +=  '{' + str(self.count) + '}'

        return out

class RegexPattern():
    def __init__(self) -> None:
        """__init__.

        :rtype: None
        """
        self.pattern = list()

    def add(self, value:Union[RegexChar, RegexCharSet,
                              RegexCharRangedSet]) -> None:
        """add.

        :param value:
        :type value: Union[RegexChar, RegexCharSet,
                                      RegexCharRangedSet]
        :rtype: None
        """
        self.pattern.append(value)

    def eact(self) -> str:
        """exact.

        :rtype: str
        """
        out = ''
        for char in self.pattern:
            out += char.exact()

        return '^' + out + '$'

    def __repr__(self) -> str:
        """__repr__.

        :rtype: str
        """
        out = ''
        for char in self.pattern:
            out += str(char)

        return '^' + out + '$'

    def __str__(self) -> str:
        """__str__.

        :rtype: str
        """
        out = ''
        for char in self.pattern:
            out += str(char)

        return '^' + out + '$'
    
    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return len(self.pattern)

    def __eq__(self, other:'RegexPattern') -> bool:
        """__eq__.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return self.exact() == other.exact()

    def equiv(self, other:'RegexPattern') -> bool:
        """equiv.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return str(self) == str(other)

