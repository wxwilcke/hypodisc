#!/usr/bin/python3

import base64
from io import BytesIO
from PIL import Image

import numpy as np

from hypodisc.data.hdf5 import UNTNDFIXL


_IMG_SIZE = 232
_IMG_CROP = 224
_IMG_MODE = "RGB"
_IMG_INTERPOLATIONMODE = Image.BILINEAR
_IMG_MEAN = np.array([0.485, 0.456, 0.406]) * 255
_IMG_STD = np.array([0.229, 0.224, 0.225]) * 255

def generate_data(g, datatypes):
    is_varlength = False
    time_dim = -1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    # maps global subject index to global subject index
    num_facts = g.triples.shape[0]
    object_to_subject = np.empty(num_facts, dtype=int)
    object_to_subject[g.triples[:, 2]] = g.triples[:, 0]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    failed = 0
    for datatype in datatypes:
        datatype_int = datatype_to_int_map[datatype]
        for g_idx in g.datatype_l2g(datatype):
            value, _ = g.i2n[g_idx]
            if value.startswith('"') or value.startswith("'"):
                # nested string
                value = value[1:-1]

            im = None
            try:
                im = b64_to_img(value)
                im = resize(im, _IMG_INTERPOLATIONMODE)
                im = centerCrop(im)
            except ValueError:
                failed += 1
                continue

            # add to matrix structures
            a = np.array(im, dtype=np.uint8)
            if _IMG_MODE == "RGB":
                # from WxHxC to CxHxW
                a = a.transpose((2, 0, 1))

            # defer normalization to runtime to reduce memory use

            # global idx of entity to which this belongs
            e_int = object_to_subject[g_idx]

            seen_datatypes.add(datatype_int)

            data[datatype_int].append(a)
            data_length[datatype_int].append(-1)
            data_entity_map[datatype_int].append(e_int)

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
                    [UNTNDFIXL for _ in seen_datatypes]))


def b64_to_img(b64string):
    im = Image.open(BytesIO(base64.urlsafe_b64decode(b64string.encode())))
    if im.mode != _IMG_MODE:
        im = im.convert(_IMG_MODE)

    return im


def resize(im, interpolation_mode=None):
    w, h = im.size
    if w == _IMG_SIZE and h == _IMG_SIZE:
        return im
    elif w == h:
        return im.resize((_IMG_SIZE, _IMG_SIZE), interpolation_mode)
    elif w > h:
        return im.resize(((_IMG_SIZE * w)//h, _IMG_SIZE), interpolation_mode)
    else:  # h < w
        return im.resize((_IMG_SIZE, (h * _IMG_SIZE)//w), interpolation_mode)


def centerCrop(im):
    w, h = im.size

    left = int(w/2 - _IMG_CROP/2)
    top = int(h/2 - _IMG_CROP/2)
    right = left + _IMG_CROP
    bottom = top + _IMG_CROP

    return im.crop((left, top, right, bottom))

def normalize(im):
    return ((im - _IMG_MEAN[:, None, None]) / _IMG_STD[:, None, None]).float()
