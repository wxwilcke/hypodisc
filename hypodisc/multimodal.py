#! /usr/bin/env python

from collections import Counter
from typing import Any, Union

import numpy as np
from hypodisc.langutil import generalize_patterns, generate_regex, RegexPattern
from rdf.terms import IRIRef, Literal
from rdf.namespaces import XSD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

def cast_values_rev(dtype:IRIRef, clusters:dict) -> list[tuple]:
    """ Cast clusters to relevant datatypes

    :param dtype:
    :type dtype: IRIRef
    :param clusters:
    :type clusters: dict
    :rtype: list[tuple]
    """
    values = list()
    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        _, freq = np.unique(clusters['assignments'], return_counts=True)
        for i in range(len(clusters)):
            mu = clusters['mu'][i]
            var = clusters['var'][i]
            n = freq[i]

            try:
                if dtype in XSD_NUMERIC:
                    if 'integer' in dtype.value.lower():
                        mu = int(mu)
                        var = int(var)

                    values.append((n, (mu, var)))
                elif dtype in XSD_DATETIME:
                    # POSIX timestamps
                    mu = cast_datetime_rev(dtype, mu)  # returns dtype
                    var = cast_datetime_delta(var)  # returns duration

                    values.append((n, (mu, var)))
                elif dtype in XSD_DATEFRAG:
                    # days
                    mu = cast_datefrag_rev(dtype, mu)  # returns dtype
                    var = cast_datefrag_delta(var)  # return dayTimeDuration

                    values.append((n, (mu, var)))
            except:
                continue
    else:  # default to string
        for pattern, n in clusters.items():
            values.append((n, pattern.exact()))

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

    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        X = X.astype(np.float32)

        num_components = range(CLUSTERS_MIN, CLUSTERS_MAX)
        clusters = compute_numeric_clusters(rng, X, num_components)
        values = cast_values_rev(dtype, clusters)
    else:  # default to string
        X = X.astype(str)

        clusters = string_clusters(X)
        values = cast_values_rev(dtype, clusters)

    return values

def compute_numeric_clusters(rng:np.random.Generator, X:np.ndarray,
                             num_components:range, num_tries:int = 3,
                             eps:float = 1e-3, standardize:bool = True,
                             shuffle:bool = True)\
                                     -> dict[str, Any]:
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
    var = np.empty(0)
    assignments = np.empty(0)
    for nc in num_components:
        bic, means, covs, y = compute_GMM(rng, X, sample, nc, num_tries, eps)
        if bic_min is None or bic + eps < bic_min:
            bic_min = bic
            mu = means
            var = covs
            assignments = y

    if standardize:
        # revert standardization
        mu = scaler.inverse_transform(mu)
        var = scaler.inverse_transform(var.squeeze(-1))

    return {'mu':mu, 'var':var, 'assignments':assignments}

def compute_GMM(rng:np.random.Generator, X:np.ndarray, sample:np.ndarray,
                num_components:int, num_tries:int,
                eps:float) -> list:
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

def string_clusters(object_list:np.ndarray, merge_charsets:bool = True,
                    omit_empty:bool = True) -> dict[RegexPattern,int]:
    """ Generate clusters of string by infering regex patterns and by
        generalizing these on similarity.

    :param object_list:
    :type object_list: np.ndarray
    :param merge_charsets:
    :type merge_charsets: bool
    :param omit_empty:
    :type omit_empty: bool
    :rtype: dict[RegexPattern,int]
    """
    patterns = list()
    for s in object_list:
        try:
            pattern = generate_regex(s)
            if not (omit_empty and len(pattern) <= 0):
                patterns.append(pattern)
        except:
            continue

    # count frequency of patterns
    patterns = dict(Counter(patterns).items())

    # merge character sets on word level and generalize these as well
    if merge_charsets:
        merged_patterns = dict()
        for p,f in patterns.items():
            q = p.generalize()
            if p == q:
                # no further generalization possible
                continue

            if q in merged_patterns.keys():
                # generalization can lead to duplicates
                merged_patterns[q] = merged_patterns[q] + f
            else:
                merged_patterns[q] = f

        # update pattern dictionary
        for p,f in merged_patterns.items():
            if p in patterns.keys():
                # sum frequencies if pattern already exists
                patterns[p] = patterns[p] + f
            else:
                patterns[p] = f

    # generalize found patterns
    for p, f in generalize_patterns(patterns).items():
        if p in patterns.keys():
            # sum frequencies if pattern already exists
            patterns[p] = patterns[p] + f
        else:
            patterns[p] = f

    return patterns
