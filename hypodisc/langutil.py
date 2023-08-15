#! /usr/bin/env python

from operator import itemgetter
from string import punctuation
from typing import Optional, Union

import numpy as np


REGEX_ILLEGAL_CHARS = ['[',']','\\','^','*','+','?','{',
                       '}','|','(',')','$','.', '"']
REGEX_WHITE_SPACE = "\s"
REGEX_LOWER = "[a-z]"
REGEX_UPPER = "[A-Z]"
REGEX_DIGIT = "[0-9]"
REGEX_PUNCT = "[(\.|\?|!)]"  # subset
REGEX_OTHER = "[^a-zA-Z0-9\.\?! ]"

class RegexCharSet():
    def __init__(self, complement:bool = False) -> None:
        """ A regular expression character set, representing a solitary
            character (eg '\\s'), a group with optional alternatives (eg
            '(a|b|c)'), or a range (eg '[a-z]').

        :param complement:
        :type complement: bool
        :rtype: None
        """
        self.complement = complement
        self._charset = np.empty(0, dtype=str)
        self._values = set()
        self._count = [[0,0]]

    def exact(self) -> str:
        """ Return a reduced character set that exactly matches the input, but
            nothing beyond it.

        :rtype: str
        """
        return str(self)

    def merge(self, other:Union['RegexCharRange',
                                'RegexCharAtom']) -> Union['RegexCharRange',
                                                           'RegexCharAtom']:
        """ Return a new character set object which encompasses both self and
            other, by merging the sets and adjusting the quantifiers.

        :param other:
        :type other: Union['RegexCharRange', 'RegexCharAtom']
        :rtype: Union['RegexCharRange','RegexCharAtom']
        """
        # assume that order matters
        charsets = [cs for cs in self._charset]
        for cs in other._charset:
            if cs not in charsets:
                charsets.append(cs)

        charsets = _collapse_charsets(charsets)
        charsets = np.array(charsets, dtype=str)
        count = [[0,0] for _ in range(len(charsets))]
        
        charset_map = {v:i for i,v in enumerate(charsets)}
        for i, charset in enumerate(self._charset):
            idx = charset_map[charset]
            count[idx][0] = self._count[i][0]
            count[idx][1] = self._count[i][1]

        for i, charset in enumerate(other._charset):
            idx = charset_map[charset]
            if count[idx][0] == 0:
                # this charset is unique to other
                count[idx][0] = other._count[i][0]
                count[idx][1] = other._count[i][1]
            else:
                # this charset is merged
                count[idx][0] = min(count[idx][0],
                                    other._count[i][0])
                count[idx][1] = max(count[idx][1],
                                    other._count[i][1])

        if type(self) is RegexCharAtom and type(other) is RegexCharAtom:
            if len(charsets) == 1:
                # same charset
                merged = RegexCharAtom(charsets[0])
            else:
                # different charset
                begin = min(ord(self.charset()[-1]), ord(other.charset()[-1]))
                end = max(ord(self.charset()[-1]), ord(other.charset()[-1]))

                charset = np.array([f'[{begin}-{end}]'], dtype=str)
                merged = RegexCharRange(charset = charset)
        else:
            merged = RegexCharRange(charset=charsets)

        merged._values = set.union(self._values, other._values)
        merged._count = count

        return merged

    def charset(self) -> str:
        """ Return the full character set without quantifiers.

        :rtype: str
        """
        out = ''
        if len(self) > 0:
            out = ''.join(self._charset)
            if type(self) is RegexCharRange:
                pre = '[' if not self.complement else '[^'
                out = pre + out + ']'

        return out

    def equiv(self, other:Union['RegexCharAtom', 'RegexCharRange']) -> bool:
        """ Return true if self and other have the same full character set,
            ignoring any quantifiers.

        :param other:
        :type other: Union['RegexCharAtom', 'RegexCharRange']
        :rtype: bool
        """
        return self.charset() == other.charset()

    def weak_match(self, other:Union['RegexCharAtom', 'RegexCharRange']) -> bool:
        """ Return true if self and other have the same reduced character set
            and quantifiers.

        :param other:
        :type other: Union['RegexCharAtom', 'RegexCharRange']
        :rtype: bool
        """
        return str(self) == str(other)

    def __repr__(self) -> str:
        """ Return a string representation of this object.

        :rtype: str
        """
        return str(self)

    def __lt__(self, other:Union['RegexCharAtom', 'RegexCharRange']) -> bool:
        """ Return true if self has a less complex character class or has less
            characters associated with it.

        :param other:
        :type other: Union['RegexCharAtom', 'RegexCharRange']
        :rtype: bool
        """
        if len(self._charset) < len(other._charset):
            return True

        if len(self._charset) == len(other._charset):
            for i in range(len(self._charset)):
                if self._charset[i] < other._charset[i]:
                    return True
            
        if self._count < other._count:
            return True

        return False

    def __eq__(self, other:Union['RegexCharAtom', 'RegexCharRange']) -> bool:
        """ Return true if self and other have the same full character set,
            including any quantifiers.

        :param other:
        :type other: Union['RegexCharAtom', 'RegexCharRange']
        :rtype: bool
        """
        return self.exact() == other.exact()

    def __len__(self) -> int:
        """ Return the shortest string length that would match the character
            set.

        :rtype: int
        """
        return sum(c[0] for c in self._count)

    def __str__(self) -> str:
        """ Print the full character class, including quantifiers.

        :rtype: str
        """
        out = ''.join(self._charset)
        if type(self) is RegexCharRange:
            pre = '[' if not self.complement else '[^'
            out = pre + out + ']'

        q_min = sum([c[0] for c in self._count])
        q_max = sum([c[1] for c in self._count])
        if q_min > 1 or q_max > q_min:
            out +=  '{' + str(q_min)
            if q_max > q_min:
                out += f',{q_max}'
            out += '}'

        return out

    def __hash__(self) -> int:
        """ Return a hash values based on the full character class, including
            quantifiers.

        :rtype: int
        """
        return hash(self.exact())

class RegexCharAtom(RegexCharSet):
    def __init__(self, charset_str:str) -> None:
        """ A regular expression character set, representing a solitary
            character (eg '\\s').

        :param value:
        :type value: str
        :rtype: None
        """
        super().__init__()

        assert len(charset_str) == 1 or (len(charset_str) == 2 and
                                         charset_str.startswith('\\'))
        self._charset = np.array([charset_str], dtype=str)
        self._count = [[0,0] for _ in range(len(self._charset))]

    def add(self, char:Optional[str]) -> None:
        """ Add character to character set count.

        :param char:
        :type char: str
        :rtype: None
        """
        self._values.add(char)
        self._count[0][0] += 1
        self._count[0][1] += 1

    def exact(self) -> str:
        """ Return true if self and other have the same reduced character set
            and quantifiers.

        :rtype: str
        """ 
        out = str(self._charset[0])
        if self._count[0][0] > 1\
            or self._count[0][1] > self._count[0][0]:
            out +=  '{' + str(self._count[0][0])
            if self._count[0][1] > self._count[0][0]:
                out += ',' + str(self._count[0][1])
            out += '}'

        return out

    def copy(self) -> 'RegexCharAtom':
        """ Return a deep copy of this character set.

        :rtype: 'RegexCharAtom'
        """
        c = RegexCharAtom(str(self._charset[0]))
        c._values = {v for v in self._values}
        c._count = [[i for i in c] for c in self._count]

        return c

class RegexCharRange(RegexCharSet):
    def __init__(self, charset:Optional[np.ndarray] = None,
                 charset_str:Optional[str] = None,
                 complement:bool = False) -> None:
        """ A regular expression character set, representing a group with
            optional alternatives (eg '(a|b|c)'), or a range (eg '[a-z]').

        :param charset:
        :type charset: Optional[np.ndarray]
        :param charset_str:
        :type charset_str: Optional[str]
        :param complement:
        :type complement: bool
        :rtype: None
        """
        super().__init__()
        self.complement = complement

        assert charset is not None or charset_str is not None
        if charset is not None:
            assert len(charset) > 0
            self._charset = charset
        elif charset_str is not None:
            assert charset_str[0] == '[' and charset_str[-1] == ']'
            self._update_charset(charset_str)

        self._count = [[0,0] for _ in range(len(self._charset))]

    def add(self, char:str) -> None:
        """ Add character to character set count.

        :param char:
        :type char: str
        :rtype: None
        """
        cs_idx = self._isidx(char)
        if cs_idx < 0:
            return

        self._values.add(char)
        self._count[cs_idx][0] += 1
        self._count[cs_idx][1] += 1

    def exact(self) -> str:
        """ Return true if self and other have the same reduced character set
            and quantifiers.

        :rtype: str
        """
        if len(self) <= 0 or self.complement:
            return str(self)

        out = ''
        for i, cs in enumerate(self._charset):
            # range [x-y]
            if '-' in cs:
                begin, end = cs.split('-')
                members = [ord(v) for v in self._values\
                           if ord(v) in range(ord(begin), ord(end) + 1)]

                if len(members) <= 0:
                    continue

                begin = chr(min(members))
                end = chr(max(members))
                if begin == end:
                    out += f'{begin}'
                else:
                    out += f'[{begin}-{end}]'

                if self._count[i][0] > 1\
                    or self._count[i][1] > self._count[i][0]:
                    out +=  '{' + str(self._count[i][0])
                    if self._count[i][1] > self._count[i][0]:
                        out += ',' + str(self._count[i][1])
                    out += '}'
            else: 
                if cs[0] == '(' and cs[-1] == ")":
                    # group (...)
                    valueset = cs[1:-1].split('|')
                else:  # fallback check
                    valueset = cs

                members = list()
                for v in self._values:
                    if v in REGEX_ILLEGAL_CHARS:
                        v = '\\' + v

                    if v in valueset:
                        members.append(v)

                if len(members) <= 0:
                    continue
                elif len(members) == 1:
                    v = members[0]
                    out += f"{v}"
                else: 
                    values = '|'.join(members)
                    out += f"({values})"
                
                if self._count[i][0] > 1\
                    or self._count[i][1] > self._count[i][0]:
                    out +=  '{' + str(self._count[i][0])
                    if self._count[i][1] > self._count[i][0]:
                        out += ',' + str(self._count[i][1])
                    out += '}'

        return out

    def copy(self) -> 'RegexCharRange':
        """ Return a deep copy of this object.

        :rtype: 'RegexCharRange'
        """
        c = RegexCharRange(charset = np.copy(self._charset))
        c._values = {v for v in self._values}
        c._count = [[i for i in c] for c in self._count]
        c.complement = self.complement

        return c

    def _update_charset(self, value:str) -> None:
        """ Parse a string representation of a character set.

        :param value:
        :type value: str
        :rtype: None
        """
        if value[1] == '^':
            self.complement = True

        charset_lst = list()
        for cs in [REGEX_LOWER, REGEX_UPPER, REGEX_DIGIT,
                   REGEX_OTHER]:
            if cs[1:-1] in value[1:-1]:
                charset_lst.append(cs[1:-1])

        if value[1] == '(' and value[-2] == ")":
            # group (...)
            charset_lst.append(value[1:-1])

        self._charset = np.array(charset_lst, dtype=str)

    def _isidx(self, char:str) -> int:
        """ Return the index of the corresponding character class if it exists,
            or -1 otherwise. 

        :param char:
        :type char: str
        :rtype: int
        """
        idx = -1  # non found code
        for i, cs in enumerate(self._charset):
            # range [x-y]
            if '-' in cs:
                assert len(cs) == 3
                begin, end = cs.split('-')
                if ord(char) in range(ord(begin), ord(end) + 1):
                    idx = i

                    break
            
            else: 
                if cs[0] == '(' and cs[-1] == ")":
                    # group (...)
                    values = cs[1:-1].split('|')
                else:  # fallback check
                    values = cs

                if char in values:
                    idx = i

                    break

        if self.complement and idx >= 0:
            # return non found code if a match is found
            return -1

        return idx

class RegexPattern():
    def __init__(self) -> None:
        """ A regular expression consisting of solitary characters (eg '\\s'),
            groups with optional alternatives (eg '(a|b|c)'), and ranges (eg
            '[a-z]').

        :rtype: None
        """
        self.pattern = list()  # type: list[Union[RegexCharAtom, RegexCharRange]]

    def add(self, charset:Union[RegexCharAtom, RegexCharRange]) -> None:
        """ Add a character set to this regular expression. Character sets are
            assumed ordered sequentially.

        :param value:
        :type value: Union[RegexCharAtom, RegexCharRange]
        :rtype: None
        """
        self.pattern.append(charset)

    def exact(self) -> str:
        """ Return a reduced pattern that exactly matches the input, but
            nothing beyond it.

        :rtype: str
        """
        out = ''
        for char in self.pattern:
            out += char.exact()

        return '^' + out + '$'

    def generalize(self) -> 'RegexPattern':
        """ Generalize character sets on word level. For example,
            '[A-Z][a-z]{2}\\s' would yield '[A-Za-z]{3}'. The generalized
            expression is returned as a new instance.

        :rtype: 'RegexPattern'
        """
        p = self.copy()
        if len(p) <= 1:
            return p

        pattern = list()
        cs_prev = p.pattern[0] if len(self.pattern) > 0 else None
        for i in range(1, len(p.pattern)):
            cs = p.pattern[i]
            if type(cs) is RegexCharRange\
                and type(cs_prev) is RegexCharRange:
                cs_prev = cs_prev.merge(cs)

                if i >= len(p.pattern) - 1:
                    # account for last member
                    pattern.append(cs_prev)

                continue

            pattern.append(cs_prev)
            if i <= 0:
                # account for first member
                pattern.append(cs_prev)
            if i >= len(p.pattern) - 1:
                # account for last member
                pattern.append(cs)

            cs_prev = cs

        p.pattern = pattern

        return p

    def copy(self) -> 'RegexPattern':
        """ Return a deep copy of this object.

        :rtype: 'RegexPattern'
        """
        p = RegexPattern()
        for cs in self.pattern:
            p.pattern.append(cs.copy())

        return p

    def weak_match(self, other:'RegexPattern') -> bool:
        """ Return a reduced regular expresion that exactly matches the input,
            but nothing beyond it.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return str(self) == str(other)
  
    def equiv(self, other:'RegexPattern') -> bool:
        """ Return true if self has the same full expression as other,
            excluding quantifiers.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return len(self) == len(other) and not False in\
                [cs.charset() == ocs.charset()\
                for cs, ocs in zip(self.pattern, other.pattern)]

    def __repr__(self) -> str:
        """ Return a string representation of this object, including
            quantifiers.

        :rtype: str
        """
        return str(self)

    def __str__(self) -> str:
        """ Return a string representation of the full expression,
            including quantifiers.

        :rtype: str
        """
        out = ''
        for char in self.pattern:
            out += str(char)

        return '^' + out + '$'
    
    def __len__(self) -> int:
        """ Return the number of unique character sets in this pattern.

        :rtype: int
        """
        return len(self.pattern)

    def __lt__(self, other:'RegexPattern') -> bool:
        """ Return true if self has a less complex pattern than other.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return len(self) < len(other) or (len(self) == len(other)\
                and True in [cs < ocs for cs, ocs in zip(self.pattern,
                                                         other.pattern)])

    def __eq__(self, other:'RegexPattern') -> bool:
        """ Return true if self has the same full expression as other,
            including quantifiers.

        :param other:
        :type other: 'RegexPattern'
        :rtype: bool
        """
        return self.exact() == other.exact()

    def __hash__(self) -> int:
        """ Return a hash value based on the full expression, including
            quantifiers. 

        :rtype: int
        """
        return hash(self.exact())

def generalize_patterns(patterns:dict[RegexPattern,set[int]],
                        num_recursions:int = 1) -> dict[RegexPattern,set[int]]:
    """ Generalize dictionary of regular expressions inplace, by merging
        similar patterns and by aligning quantifiers.

    :param patterns:
    :type patterns: dict[RegexPattern,int]
    :param num_recursions:
    :type num_recursions: int
    :rtype: dict[RegexPattern,int]
    """
    if len(patterns) <= 0:
        return patterns

    generalized_patterns = dict()  #type: dict[RegexPattern, set[int]]
    for l in {len(p) for p in patterns.keys()}:
        # pattern with same number of subpatterns
        subset = {p:m for p,m in patterns.items() if len(p) == l}
        for p, members in _generalize_patterns_uniform(subset).items():
            if p not in generalized_patterns.keys():
                generalized_patterns[p] = set()

            generalized_patterns[p] = generalized_patterns[p].union(members)

    if num_recursions > 0:
        nrec = num_recursions - 1
        for p, members in generalize_patterns(generalized_patterns,
                                              num_recursions = nrec).items():
            # don't sum here since that is already done in the called function
            generalized_patterns[p] = members

    return generalized_patterns

def _generalize_patterns_uniform(patterns:dict[RegexPattern,set[int]])\
        -> dict[RegexPattern,set[int]]:
    """ Generalize patterns of same length by updating quantifiers.

    :param patterns:
    :type patterns: dict[RegexPattern,int]
    :rtype: dict[RegexPattern,int]
    """
    if len(patterns) <= 1:
        # nothing to merge
        return dict()

    # deterministic order
    i2p = list(patterns.keys())

    # match on unquantified character (set)
    pattern_mat = np.array([[cs.charset() for cs in p.pattern]
                            for p in i2p], dtype=str)
    eq_mat = pattern_mat[:, None] == pattern_mat  # elementwise matching

    # find matching patterns
    matches = set()
    nonmatches = list()
    for i in range(eq_mat.shape[0]):
        match_idx = np.where(eq_mat[i].all(axis = -1))[0]
        if len(match_idx) >= 2:
            # two or more patterns are similar
            matches.add(tuple(match_idx))

            continue

        nonmatches.append(i)

    # merge matching patterns
    out = dict()  #type: dict[RegexPattern, set[int]]
    for match_idx in matches:
        siblings = list(itemgetter(*match_idx)(i2p))
        merged_pattern = _merge_patterns(siblings)
        members = set.union(*[patterns[i2p[i]] for i in match_idx])

        if merged_pattern not in patterns.keys():
            out[merged_pattern] = members

            continue
        
        out[merged_pattern] = patterns[merged_pattern].union(members)

    return out

def _merge_patterns(patterns:list[RegexPattern]) -> RegexPattern:
    """ Merge two or more patterns

    :param patterns:
    :type patterns: list[RegexPattern]
    :rtype: RegexPattern
    """
    num_charsets = len(patterns[0])
    merged_pattern = RegexPattern()
    for i in range(num_charsets):
        # merge character sets
        charsets = [rp.pattern[i] for rp in patterns]
        merged_cs = _merge_charsets(charsets)
        merged_pattern.add(merged_cs)

    return merged_pattern

def _merge_charsets(charsets:list) -> Union[RegexCharAtom, RegexCharRange]:
    """ Merge two or more character sets

    :param charsets:
    :type charsets: list
    :rtype: Union[RegexCharAtom, RegexCharRange]
    """
    charset = charsets[0]
    for charset_next in charsets[1:]:
        charset = charset.merge(charset_next)

    return charset

def generate_regex(s:str, strip_punctuation:bool = True) -> RegexPattern:
    """ Generate regular expresion that fits string. 

    :param s:
    :type s: str
    :param strip_punctuation:
    :type strip_punctuation: bool
    :rtype: RegexPattern
    """
    # remove unecessary white space and punctuation
    s = ' '.join(s.split())
    if strip_punctuation:
        s.translate(str.maketrans('', '', punctuation))
    slen = len(s)

    pattern = RegexPattern()
    if slen <= 0:
        # empty string
        return pattern

    char_set = _mkcharset(_char_to_regex(s[0]))
    char_set.add(s[0])
    for i in range(1, slen):
        symbol_char_set = _char_to_regex(s[i])
        if symbol_char_set == char_set.charset():
            # pattern continues still
            char_set.add(s[i])
        
            # account for final character
            if i >= slen - 1:
                pattern.add(char_set)

            continue

        # add interupted pattern to output
        pattern.add(char_set)

        # start next char set
        char_set = _mkcharset(symbol_char_set)
        char_set.add(s[i])

        # account for final character
        if i >= slen - 1:
            pattern.add(char_set)

    return pattern

def _mkcharset(char_set:str) -> Union[RegexCharAtom, RegexCharRange]:
    """ Wrap a character set string in a dedicated object.

    :param char_set:
    :type char_set: str
    :rtype: Union[RegexCharAtom, RegexCharRange]
    """
    if char_set in {REGEX_DIGIT, REGEX_LOWER, REGEX_UPPER, REGEX_PUNCT}:
        char_set_object = RegexCharRange(charset_str = char_set)
    else:  # white space and other non-word chars
        char_set_object = RegexCharAtom(char_set)

    return char_set_object

def _char_to_regex(c:str) -> str:
    """ Return suitable character class or (escaped) char if none fit.

    :param c:
    :type c: str
    :rtype: str
    """
    char_class = _character_class(c)
    if char_class == REGEX_OTHER:
        if c in REGEX_ILLEGAL_CHARS:
            # escape char
            c = "\\" + c

        return c

    return char_class

def _character_class(c:str) -> str:
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

def _inrange(charset:str, char:str, complement:bool = False) -> bool:
    """ Return true if character is part of a character set, or false if
        the complement is asked.

    :param charset:
    :type charset: str
    :param char:
    :type char: str
    :param complement:
    :type complement: bool
    :rtype: bool

    """
    isin = False
    if charset[0] == '[' and charset[-1] == ']':
        charset = charset[1:-1]

    # range [x-y]
    if '-' in charset:
        assert len(charset) == 3
        begin, end = charset.split('-')
        if ord(char) in range(ord(begin), ord(end) + 1):
            isin = True        
    else: 
        if charset[0] == '(' and charset[-1] == ")":
            # group (...)
            values = charset[1:-1].split('|')
        else:  # fallback check
            values = charset

        if char in values:
            isin = True        

    if complement:
        isin = not isin

    return isin

def _collapse_charsets(charsets:list[str]) -> list[str]:
    """ Remove unecessary character sets, by removing characters which are
        already represented by a character set.

    :param charsets:
    :type charsets: list[str]
    :rtype: list[str]
    """
    out = list()
    for i in range(len(charsets)):
        cs = charsets[i]
        if not(len(cs) == 1 or len(cs) == 2 and cs.startswith('\\')):
            # not a character
            out.append(cs)

            continue

        is_contained = False
        for j in range(len(charsets)):
            if i == j:
                continue

            if _inrange(charsets[j], cs[-1]):
                is_contained = True

                break

        if not is_contained:
            out.append(cs)

    return out

