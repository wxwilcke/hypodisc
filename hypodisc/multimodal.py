#! /usr/bin/env python

from collections import Counter

import numpy as np
from hypodisc.langutil import generalize_regex, generate_regex
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


def string_clusters(object_list:np.ndarray, omit_empty:bool = True) -> list:
    """ Generate clusters of string by infering regex patterns and by
        generalizing these on similarity.

    :param object_list:
    :type object_list: np.ndarray
    :param omit_empty:
    :type omit_empty: bool
    :rtype: list
    """
    patterns_all = list()
    for s in object_list:
        try:
            pattern = generate_regex(s)
            if not (omit_empty and len(pattern) <= 0):
                patterns_all.append(pattern)
        except:
            continue

    # save those occurring more than once
    common_patterns = list()
    for pattern, freq in Counter(patterns_all).items():
        if freq > 1:
            common_patterns.append((pattern, freq))

    # generalize unique patterns
    for pattern, freq in generalize_regex(patterns_all):
        # TODO: merge freq above with freq here
        if freq > 1:
            common_patterns.append((pattern, freq))

    return common_patterns
