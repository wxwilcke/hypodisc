#!/usr/bin/python3

import numpy as np

import torch

from hypodisc.data.hdf5 import INT2DVARL


# textual
_STR_MAX_CHARS = 512
_TOKENIZER_CONFIG = ["huggingface/pytorch-transformers",
                     "tokenizer"]
_TOKENIZER_PAD_TOKEN = "[PAD]"

def generate_data(g, datatypes, weights="distilbert-base-multilingual-cased"):
    is_varlength = True
    time_dim = 1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    # maps global object index to global subject index
    num_facts = g.triples.shape[0]
    object_to_subject = np.empty(num_facts, dtype=int)
    object_to_subject[g.triples[:, 2]] = g.triples[:, 0]

    tokenizer = _loadFromHub(_TOKENIZER_CONFIG, weights)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': _TOKENIZER_PAD_TOKEN})

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    failed = 0
    for datatype in datatypes:
        datatype_int = datatype_to_int_map[datatype]
        for g_idx in g.datatype_l2g(datatype):
            value, _ = g.i2n[g_idx]

            sequence = None
            try:
                value = str(value)
                sequence = encode(tokenizer, value)
            
                a = np.array(sequence, dtype=np.int32)[:_STR_MAX_CHARS]
                a = a[np.newaxis, ...]
            except ValueError:
                failed += 1

                continue

            seq_length = a.shape[1]
            if seq_length <= 0:
                failed += 1

                continue

            # global idx of entity to which this belongs
            e_int = object_to_subject[g_idx]

            data[datatype_int].append(a)
            data_length[datatype_int].append(seq_length)
            data_entity_map[datatype_int].append(e_int)
            seen_datatypes.add(datatype_int)

    if failed > 0:
        print(f" ({failed} failed) ", end='')

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes],
                    [INT2DVARL for _ in seen_datatypes]))

def encode(tokenizer, sentence):
    return tokenizer.encode(sentence, add_special_tokens=True)

def _loadFromHub(parameters, weights=None):
    return torch.hub.load(*parameters, weights)
