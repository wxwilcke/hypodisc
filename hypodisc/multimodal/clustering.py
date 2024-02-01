#! /usr/bin/env python

from sys import maxsize
from typing import Any

import numpy as np
from rdf.terms import IRIRef, Literal
from rdf.namespaces import XSD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from hypodisc.multimodal.datatypes import (XSD_DATEFRAG, XSD_DATETIME,
                                           XSD_NUMERIC, XSD_STRING)
from hypodisc.multimodal.langutil import (generalize_patterns, generate_regex,
                                          RegexPattern)
from hypodisc.multimodal.timeutils import (cast_datefrag_delta,
                                           cast_datefrag_rev, cast_datefrag,
                                           cast_datetime, cast_datetime_delta,
                                           cast_datetime_rev)


SUPPORTED_XSD_TYPES = set.union(XSD_DATEFRAG,
                                XSD_DATETIME,
                                XSD_NUMERIC,
                                XSD_STRING)

CLUSTERS_MIN = 1
CLUSTERS_MAX = 10
SEED_MIN = 0
SEED_MAX = 2**32 - 1


def cast_values(dtype:IRIRef, values:list) -> tuple[np.ndarray, np.ndarray]:
    """ Cast raw values to a datatype suitable for clustering.
        Default to string.

    :param dtype:
    :type dtype: IRIRef
    :param values:
    :type values: list
    :rtype: np.ndarray
    """
    X = np.empty(len(values), dtype=object)
    X_idx = list()

    if dtype in XSD_NUMERIC:
        func = lambda _, v : float(v)
    elif dtype in XSD_DATETIME:
        # cluster on POSIX timestamps
        func = lambda dtype, v : cast_datetime(dtype, v)
    elif dtype in XSD_DATEFRAG:
        # cluster on days
        func = lambda dtype, v : cast_datefrag(dtype, v)
    else:  # default to XSD_STRING:
        # nothing changes
        func = lambda _, v : str(v)

    for i, v in enumerate(values):
        try:
            X[i] = func(dtype, v)
        except:
            continue
    
        X_idx.append(i)

    return X[X_idx], np.array(X_idx, dtype=int)

def cast_values_rev_dist(dtype:IRIRef, clusters:list[tuple])\
        -> list[tuple[set, Any]]:
    """ Cast clusters to relevant datatypes distributions

    :param dtype:
    :type dtype: IRIRef
    :param clusters:
    :type clusters: list[tuple]
    :rtype: list[tuple[set, Any]]
    """
    values = list()
    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        for mu, sigma, members in clusters:
            try:
                if dtype in XSD_NUMERIC:
                    if 'integer' in dtype.value.lower():
                        mu = int(mu)
                        sigma = int(sigma)

                    values.append((members, (mu, sigma)))
                elif dtype in XSD_DATETIME:
                    # POSIX timestamps
                    mu = cast_datetime_rev(dtype, mu)  # returns dtype
                    sigma = cast_datetime_delta(sigma)  # returns duration

                    values.append((members, (mu, sigma)))
                elif dtype in XSD_DATEFRAG:
                    # days
                    mu = cast_datefrag_rev(dtype, mu)  # returns dtype
                    sigma = cast_datefrag_delta(sigma)  # return dayTimeDuration

                    values.append((members, (mu, sigma)))
            except:
                continue
    else:  # default to string
        for pattern, members in clusters:
            values.append((members, pattern.exact()))

    return values

def cast_values_rev(dtype:IRIRef, clusters:list[tuple])\
        -> list[tuple[set, Any]]:
    """ Cast clusters to relevant datatypes ranges

    :param dtype:
    :type dtype: IRIRef
    :param clusters:
    :type clusters: list[tuple]
    :rtype: list[tuple[set, Any]]
    """
    values = list()
    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        for mu, sigma, members in clusters:
            try:
                lower_bound = mu - 3 * sigma
                upper_bound = mu + 3 * sigma
                if dtype in XSD_NUMERIC:
                    if 'integer' in dtype.value.lower():
                        lower_bound = int(lower_bound)
                        upper_bound = int(upper_bound)
                    else:
                        lower_bound = float(lower_bound)
                        upper_bound = float(upper_bound)

                    values.append((members, (lower_bound, upper_bound)))
                elif dtype in XSD_DATETIME:
                    # POSIX timestamps
                    lower_bound = cast_datetime_rev(dtype, lower_bound)  # returns dtype
                    upper_bound = cast_datetime_rev(dtype, upper_bound)  # returns dtype

                    values.append((members, (lower_bound, upper_bound)))
                elif dtype in XSD_DATEFRAG:
                    # days
                    lower_bound = cast_datefrag_rev(dtype, lower_bound)  # returns dtype
                    upper_bound = cast_datefrag_rev(dtype, upper_bound)  # returns dtype

                    values.append((members, (lower_bound, upper_bound)))
            except:
                continue
    else:  # default to string
        for pattern, members in clusters:
            values.append((members, pattern.exact()))

    return values

def compute_clusters(rng:np.random.Generator, dtype:IRIRef,
                     values:list, values_gidx:np.ndarray)\
                             -> list[tuple[set, Any]]:
    """Compute clusters from list of values.

    :param rng:
    :type rng: np.random.Generator
    :param dtype:
    :type dtype: IRIRef
    :param values:
    :type values: list[Literal]
    :param values_gidx:
    :type values_gidx: np.ndarray
    :rtype: list[tuple[set, Any]]
    """

    X, X_idx = cast_values(dtype, values)
    X_gidx = values_gidx[X_idx]  # global indices of nodes in order of X

    if dtype in set.union(XSD_NUMERIC, XSD_DATETIME, XSD_DATEFRAG):
        X = X.astype(np.float32)

        num_components = range(CLUSTERS_MIN, CLUSTERS_MAX)
        means, stdevs, assignments = compute_numeric_clusters(rng, X,
                                                              num_components)
        clusters = [(means[i], stdevs[i], set(X_gidx[assignments == i]))
                    for i in range(len(means))]
        values = cast_values_rev(dtype, clusters)
    else:  # default to string
        X = X.astype(str)

        clusters = string_clusters(X, X_gidx)
        values = cast_values_rev(dtype, clusters)

    return values

def compute_numeric_clusters(rng:np.random.Generator, X:np.ndarray,
                             num_components:range, num_tries:int = 3,
                             eps:float = 1e-3, standardize:bool = True,
                             shuffle:bool = True)\
                                -> tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray]:
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
    covar = np.empty(0)
    assignments = np.empty(0)
    for nc in num_components:
        bic, means, covars, y = compute_GMM(rng, X, sample, nc, num_tries, eps)
        if bic_min is None or bic + eps < bic_min:
            bic_min = bic
            mu = means
            covar = covars
            assignments = y

    if standardize:
        # revert standardization
        mu = scaler.inverse_transform(mu)
        sigma = np.einsum('s,pqr->psq', np.sqrt(scaler.var_), covar)
    else:
        # compute standard deviations
        num_components = mu.shape[0]
        sigma = np.array([[np.sqrt(np.trace(covar[i])/num_components)]
                         for i in range(0, num_components)])

    return mu, sigma, assignments

def compute_GMM(rng:np.random.Generator, X:np.ndarray, sample:np.ndarray,
                num_components:int, num_tries:int,
                eps:float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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


    bic_min = float(maxsize)  # best score
    mu = np.empty(0)
    covar = np.empty(0)
    assignments = np.empty(0)
    for _ in range(num_tries):
        seed = rng.integers(SEED_MIN, SEED_MAX)
        gmm = GaussianMixture(n_components = num_components,
                              random_state = seed)
        gmm.fit(sample)

        bic = gmm.bic(sample)
        if bic_min is None or bic + eps < bic_min:
            bic_min = float(bic)
            mu = gmm.means_
            covar = gmm.covariances_

            assignments = gmm.predict(X)

    return bic_min, mu, covar, assignments

def string_clusters(X:np.ndarray, X_gidx:np.ndarray,
                    merge_charsets:bool = True, omit_empty:bool = True)\
                            -> list[tuple[RegexPattern,set[int]]]:
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
    patterns = dict()  #type: dict[RegexPattern, set[int]]
    for i, s in enumerate(X):
        try:
            pattern = generate_regex(s)
        except:
            continue
            
        if not (omit_empty and len(pattern) <= 0):
            # map patterns to global node indices whose value they match
            idx_set = { X_gidx[i] }
            if pattern not in patterns.keys():
                patterns[pattern] = set()

            patterns[pattern] = patterns[pattern].union(idx_set)

    # merge character sets on word level and generalize these as well
    if merge_charsets:
        merged_patterns = dict()  #type: dict[RegexPattern, set[int]]
        for p, members in patterns.items():
            q = p.generalize()
            if p == q:
                # no further generalization possible
                continue

            if q not in merged_patterns.keys():
                merged_patterns[q] = set()

            merged_patterns[q] = merged_patterns[q].union(members)

        # update pattern dictionary
        for p, members in merged_patterns.items():
            if p not in patterns.keys():
                patterns[p] = set()

            patterns[p] = patterns[p].union(members)

    # generalize found patterns
    for p, members in generalize_patterns(patterns).items():
        if p not in patterns.keys():
            patterns[p] = set()

        patterns[p] = patterns[p].union(members)

    return list(patterns.items())
