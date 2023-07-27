#! /usr/bin/env python

from collections import Counter
from datetime import datetime
from random import randint
from typing import cast

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from rdf.terms import IRIRef, Literal
from rdf.namespaces import XSD

from hypodisc.timeutils import (cast_datefrag_delta, cast_datefrag_rev, cast_datefrag,
                                cast_datetime, cast_datetime_delta, cast_datetime_rev)


XSD_DATEFRAG = {XSD + 'gDay', XSD + 'gMonth', XSD + 'gMonthDay'}
XSD_DATETIME = {XSD + 'date', XSD + 'dateTime', XSD + 'dateTimeStamp',
                XSD + 'gYear', XSD + 'gYearMonth'}
XSD_NUMERIC = {XSD + 'integer', XSD + 'nonNegativeInteger',
               XSD + 'positiveInteger', XSD + 'float', XSD + 'decimal',
               XSD + 'double', XSD + 'negativeInteger',
               XSD + 'nonPositiveInteger'}
XSD_STRING = {XSD + 'string', XSD + 'normalizedString'}
SUPPORTED_XSD_TYPES = set.union(XSD_DATEFRAG,
                                XSD_DATETIME,
                                XSD_NUMERIC,
                                XSD_STRING)

CLUSTERS_MIN = 1
CLUSTERS_MAX = 10
SEED_MIN = 0
SEED_MAX = (2**31) - 1


def cast_values(dtype:IRIRef, values:list) -> np.ndarray:
    """ Cast raw values to a datatype suitable for clustering.
        Default to string.

    :param dtype:
    :type dtype: IRIRef
    :param values:
    :type values: list
    :rtype: np.ndarray
    """
    X = np.empty(len(values), dtype=object)
    i = 0
    for v in values:
        try:
            if dtype in XSD_NUMERIC:
                v = float(v)
            elif dtype in XSD_DATETIME:
                # cluster on POSIX timestamps
                v = cast_datetime(dtype, v)
            elif dtype in XSD_DATEFRAG:
                # cluster on days
                v = cast_datefrag(dtype, v)
            else:  # default to XSD_STRING:
                # nothing changes
                v = str(v)
        except:
            continue
    
        X[i] = v
        i += 1

    return X[:i]

def cast_values_rev(dtype:IRIRef, clusters:dict) -> list:
    """ Cast clusters to relevant datatypes

    :param dtype:
    :type dtype: IRIRef
    :param clusters:
    :type clusters: dict
    :rtype: list
    """
    _, freq = np.unique(clusters['assignments'], return_counts=True)
    
    values = list()
    for i in range(len(clusters)):
        mu = clusters['mu'][i]
        sigma = clusters['sigma'][i]
        n = freq[i]

        try:
            if dtype in XSD_NUMERIC:
                if 'integer' in dtype.value.lower():
                    mu = int(mu)
                    sigma = int(sigma)

                values.append((n, mu, sigma))
            elif dtype in XSD_DATETIME:
                # POSIX timestamps
                mu = cast_datetime_rev(dtype, mu)  # returns dtype
                sigma = cast_datetime_delta(sigma)  # returns duration

                values.append((n, mu, sigma))
            elif dtype in XSD_DATEFRAG:
                # days
                mu = cast_datefrag_rev(dtype, mu)  # returns dtype
                sigma = cast_datefrag_delta(sigma)  # return dayTimeDuration

                values.append((n, mu, sigma))
            else:  # default to XSD_STRING
                pass

        except:
            continue

    return values

def compute_clusters(rng:np.random.Generator, dtype:IRIRef,
                     values:list) -> list:
    """Compute clusters from list of values.

    :param dtype:
    :type dtype: IRIRef
    :param values:
    :type values: list
    :rtype: list
    """
    X = cast_values(dtype, values)

    num_components = range(CLUSTERS_MIN, CLUSTERS_MAX)
    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        X = X.astype(np.float32)
        clusters = compute_numeric_clusters(rng, X, num_components)
        values = cast_values_rev(dtype, clusters)
    else:  # default to string
        clusters = string_clusters(values)

    return values

def compute_numeric_clusters(rng:np.random.Generator, X:np.ndarray,
                             num_components:range, num_tries:int = 3,
                             eps:float = 1e-3, standardize:bool = True,
                             shuffle:bool = True) -> dict:
    """ Compute numerical cluster means and stdevs for a range of possible
        number of components. Also return the cluster assignments.

    :param rng:
    :type rng: np.random.Generator
    :param X:
    :type X: np.ndarray
    :param num_components:
    :type num_components: range
    :param num_tries:
    :type num_tries: int
    :param eps:
    :type eps: float
    :param standardize:
    :type standardize: bool
    :param shuffle:
    :type shuffle: bool
    :rtype: dict
    """
    if X.ndim == 1:
        # convert to array of shape (n_samples, n_features)
        X = X.reshape(-1, 1)

    scaler = StandardScaler()
    if standardize:
        # standardize
        scaler.fit(X)
        X = scaler.transform(X)

    # compensate for small datasets
    # let stdev of noise scale with data
    sample = X
    stdev = np.sqrt(X.var())
    while sample.shape[0] < 1024:
        sample = np.vstack([sample,
                            X + rng.normal(0, stdev/(stdev+1))])

    if shuffle:
        # shuffle order
        rng.shuffle(sample)

    bic_min = None  # best score
    mu = np.empty(0)
    sigma = np.empty(0)
    assignments = np.empty(0)
    for nc in num_components:
        bic, means, covs, y = compute_GMM(rng, X, sample, nc, num_tries, eps)
        if bic_min is None or bic + eps < bic_min:
            bic_min = bic
            mu = means
            sigma = covs
            assignments = y

    if standardize:
        # revert standardization
        mu = scaler.inverse_transform(mu)
        sigma = scaler.inverse_transform(sigma.squeeze(-1))

    sigma = np.sqrt(sigma)  # stdev

    return {'mu':mu, 'sigma':sigma, 'assignments':assignments}

def compute_GMM(rng:np.random.Generator, X:np.ndarray, sample:np.ndarray,
                num_components:int, num_tries:int, eps:float) -> list:
    """ Compute a GMM from different random states and return the best results.
        Train the model on the sample but returns the predictions on X

    :param rng:
    :type rng: np.random.Generator
    :param X:
    :type X: np.ndarray
    :param num_components:
    :type num_components: int
    :param num_tries:
    :type num_tries: int
    :param eps:
    :type eps: float
    :rtype: list
    """


    bic_min = None  # best score
    mu = np.empty(0)
    sigma = np.empty(0)
    assignments = np.empty(0)
    for _ in range(num_tries):
        seed = rng.integers(SEED_MIN, SEED_MAX)
        gmm = GaussianMixture(n_components = num_components,
                              random_state = seed)
        gmm.fit(sample)

        bic = gmm.bic(sample)
        if bic_min is None or bic + eps < bic_min:
            bic_min = bic
            mu = gmm.means_
            sigma = gmm.covariances_

            assignments = gmm.predict(X)

    return [bic_min, mu, sigma, assignments]


def string_clusters(object_list, strict=True):
    regex_patterns = list()
    for s in object_list:
        regex_patterns.append(generate_regex(s))

    if not strict:
        return list(generalize_regex(regex_patterns))

    weighted = {k:v**2 for k,v in Counter(regex_patterns).items()}

    wmin = min(weighted.values())
    wmax = max(weighted.values())
    if wmin == wmax:
        return regex_patterns

    weighted_normalized = {k:(v-wmin)/(wmax-wmin) for k,v in weighted.items()}

    return [pattern for pattern in weighted_normalized.keys() if
            weighted_normalized[pattern] >= NORMALIZED_MIN]

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

def generate_regex(s):
    s = ' '.join(s.split())

    pattern = '^'
    if len(s) <= 0:
        # empty string
        return pattern + '$'

    prev_char_class = character_class(s[0])
    count = 0
    for i in range(len(s)):
        char_class = character_class(s[i])

        if char_class == prev_char_class:
            count += 1

            if i < len(s)-1:
                continue

        pattern += prev_char_class
        if prev_char_class != "\s":
            pattern += '{' + str(count) + '}'
        count = 1
        if i >= len(s)-1 and char_class != prev_char_class:
            pattern += char_class
            if char_class != "\s":
                pattern += '{' + str(count) + '}'

        prev_char_class = char_class

    return pattern + '$'

def character_class(c):
    if c.isalpha():
        char_class = "[a-z]" if c.islower() else "[A-Z]"
    elif c.isdigit():
        char_class = "\d"
    elif c.isspace():
        char_class = "\s"
    elif c == "." or c == "?" or c == "!":
        char_class = "[(\.|\?|!)]"
    else:
        char_class = "[^A-Za-z0-9\.\?! ]"

    return char_class
