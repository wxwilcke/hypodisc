#!/usr/bin/env python

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as f
try:
    from bhtsne.bhtsne import run_bh_tsne
    print("[AVAIL] Module t-SNE detected")
except ModuleNotFoundError:
    pass


def compute_clusters(model, features, epoch, out_dir, featureless, devices, 
                     **kwargs):
    encoders, distmult = model
    encoder_device, decoder_device = devices

    X, E_idx = features
    node_embeddings = distmult.node_embeddings[E_idx, :]
    if not featureless:
        # compute embeddings over all entities
        # TODO: compute in batches?
        features = [X, E_idx, encoder_device]
        with torch.no_grad():
            feature_embeddings = encoders(features).to(decoder_device)

            # fuse entity and literal embeddings
            node_embeddings = distmult.fuse_model([node_embeddings,
                                                   feature_embeddings])

    node_embeddings = node_embeddings.detach().numpy()
    node_embeddings = ((node_embeddings - np.mean(node_embeddings))
                       / np.std(node_embeddings))  # standardize

    # TODO: add sampling parameter?
    clusters = run_bh_tsne(node_embeddings,
                           initial_dims=node_embeddings.shape[1],
                           **kwargs)

    filename = f"node_embedding_clusters_{epoch}_epoch.gz"
    np.savetxt(out_dir+filename, X=clusters)

def getConfParam(config, name, default=None):
    """ Access nested dict by dot-separated key strings
    """
    names = name.split('.')
    for name in names:
        if not isinstance(config, dict) or name not in config.keys():
            return default

        config = config[name]

    return config

def mkbatches(sample_idx, batch_size=1):
    """ split array in batches
    """
    n = len(sample_idx)  # number of samples
    batch_size = min(n, batch_size)
    idc = np.arange(n, dtype=np.int32)

    idc_assignments = np.array_split(idc, n/batch_size)
    sample_assignments = [np.array(sample_idx, dtype=np.int32)[slce]
                          for slce in idc_assignments]

    return list(zip(idc_assignments, sample_assignments))


def mkbatches_varlength(sample_idx, seq_length_map,
                        batch_size=1):
    n = len(sample_idx)
    batch_size = min(n, batch_size)

    # sort on length
    idc = np.arange(n, dtype=np.int32)
    _, sequences_sorted_idc = zip(*sorted(zip(seq_length_map, idc)))

    seq_assignments = np.array_split(sequences_sorted_idc, n/batch_size)
    sample_assignments = [np.array(sample_idx, dtype=np.int32)[slce]
                          for slce in seq_assignments]

    return list(zip(seq_assignments, sample_assignments))


def zero_pad(samples, time_dim, min_length=-1):
    if time_dim < 0:
        return samples

    max_height = max([t.shape[0] for t in samples])
    max_width = max([t.shape[1] for t in samples])

    if min_length > 0:
        if time_dim == 0:
            max_height = max(max_height, min_length)
        elif time_dim == 1:
            max_width = max(max_width, min_length)

    return [f.pad(t, [0, max_width-t.shape[1], 0, max_height-t.shape[0]])
            for t in samples]


def mksplits(entity_to_class_map, splits="60/20/20"):
    _, test_split, valid_split = [int(i) for i in splits.split('/')]

    classes, counts = np.unique(entity_to_class_map, return_counts=True)
    train, test, valid = list(), list(), list()
    for i, c in enumerate(classes):
        num_test = round(counts[i] * (splits[1]/100))
        num_train = counts[i] - num_test
        num_valid = round(num_train * (splits[2]/100))
        num_train -= num_valid

        class_idx = np.where(entity_to_class_map == c)[0]
        np.random.shuffle(class_idx)

        train.extend(class_idx[:num_train])
        valid.extend(class_idx[num_train:(num_train+num_valid)])
        test.extend(class_idx[-num_test:])

    return (train, test, valid)


def categorical_accuracy(Y_hat, Y_ground):
    predictions = Y_hat.max(axis=1)[1]
    num_correct = sum(predictions == Y_ground)

    return num_correct / float(len(Y_ground))

def global_to_local(data, global_index):
    # remap global indices to local indices of embeddings
    # assume that embeddings are ordered by global_index
    global_to_local_map = torch.empty(max(global_index) + 1, dtype=int)
    global_to_local_map[global_index] = torch.arange(len(global_index))

    data[:, 0] = global_to_local_map[data[:, 0]]
    data[:, 2] = global_to_local_map[data[:, 2]]

    return data


def add_noise_(encoding_sets, p_noise, multiplier=0.01):
    if p_noise <= 0:
        return

    for mset in encoding_sets:
        data = mset[1]
        for i in range(len(data)):
            if isinstance(data[i], np.ndarray):
                size = data[i].size
                shape = data[i].shape

                b = np.random.binomial(1, p_noise, size=size)
                noise = b.reshape(shape) * (2 * np.random.random(shape) - 1)
                data[i] += multiplier * noise
            elif isinstance(data[i], list):
                size = len(data[i])

                b = np.random.binomial(1, p_noise, size=size)
                noise = b * (2 * np.random.random(size) - 1)
                data[i] = list(data[i] + multiplier * noise)
            elif isinstance(data[i], float):
                if np.random.random() < p_noise:
                    continue

                noise = float(2 * np.random.random() - 1)
                data[i] += multiplier * noise
            else:
                raise Exception(f" Cannot add noise to {type(data[i])}")

class EarlyStop:
    stop = None
    tolerance = -1
    patience = -1
    _patience_default = -1
    best_score = -1
    best_weights = None
    best_optim = None

    def __init__(self, patience=7, tolerance=0.001):
        self.tolerance = tolerance
        self._patience_default = patience

        self.reset_counter()

    def record(self, score, weights, optim):
        if self.best_score < 0:
            self._update(score, weights, optim)

            return

        self.patience -= 1
        if (score + self.tolerance) < self.best_score:
            self._update(score, weights, optim)
            self.reset_counter()

        if self.patience <= 0:
            self.stop = True

    def _update(self, score, weights, optim):
        self.best_score = score
        self.best_weights = deepcopy(weights.state_dict())
        self.best_optim = deepcopy(optim.state_dict())

    def reset_counter(self):
        self.patience = self._patience_default
        self.stop = False
