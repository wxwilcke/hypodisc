#!/usr/bin/python3

from deep_geometry import vectorizer as gv
import numpy as np

from hypodisc.data.hdf5 import FLT2DVARL


_MAX_POINTS = 64
VEC_LENGTH = 9


def generate_data(g, datatypes, time_dim=1):
    is_varlength = True

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
            value.strip()

            vec = None
            try:
                value = str(value)
                if value.startswith('"') or value.startswith("'"):
                    # nested string
                    value = value[1:-1]
                vec = gv.vectorize_wkt(value)[:_MAX_POINTS, :]
            except ValueError:
                failed += 1
                continue

            vec_length = vec.shape[0]
            if vec_length <= 0:
                continue

            # add means of X,Y to vector
            mean_x = np.mean(vec[:, 0])
            mean_y = np.mean(vec[:, 1])
            vec = np.hstack([np.vstack([[mean_x, mean_y]]*vec_length), vec])

            if time_dim == 1:
                # prep for CNN
                vec = vec.T

            # global idx of entity to which this belongs
            e_int = object_to_subject[g_idx]

            seen_datatypes.add(datatype_int)
            data[datatype_int].append(vec)
            data_length[datatype_int].append(vec.shape[time_dim])
            data_entity_map[datatype_int].append(e_int)

    if failed > 0:
        print(f" ({failed} failed) ", end='')

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    # normalize
    for i in range(len(data)):
        a = data[i]

        sc = GeomScaler(time_dim=time_dim)
        means = sc.fit(a)
        data[i] = sc.transform(a, means)

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes],
                    [FLT2DVARL for _ in seen_datatypes]))


class GeomScaler:
    FULL_STOP_INDEX = -1

    def __init__(self, time_dim=0):
        self.scale_factor = 1.
        self.time_dim = time_dim

    def fit(self, geometry_vectors):
        means = [self.localized_mean(v) for v in geometry_vectors]
        if self.time_dim == 1:
            means = [mean[np.newaxis].T for mean in means]

        min_maxs = list()
        for index, geometry in enumerate(geometry_vectors):
            full_stop_point_index = self.get_full_stop_index(geometry)

            x_and_y_coords = geometry[:full_stop_point_index, 2:4]\
                if self.time_dim == 0\
                else geometry[2:4, :full_stop_point_index]
            min_maxs.append([
                np.min(x_and_y_coords - means[self.time_dim]),
                np.max(x_and_y_coords - means[self.time_dim])
            ])

        self.scale_factor = np.std(min_maxs)

        return means

    def transform(self, geometry_vectors, means):
        localized = list()
        for index, geometry in enumerate(geometry_vectors):
            stop_index = self.get_full_stop_index(geometry) + 1
            geometry_copy = geometry.copy()
            if self.time_dim == 0:
                geometry_copy[:stop_index, 2:4] -= means[index]
                geometry_copy[:stop_index, 2:4] /= self.scale_factor
            else:
                geometry_copy[2:4, :stop_index] -= means[index]
                geometry_copy[2:4, :stop_index] /= self.scale_factor

            localized.append(geometry_copy)

        return localized

    def get_full_stop_index(self, geometry_vector):
        full_stop_slice = geometry_vector[:, self.FULL_STOP_INDEX]\
                if self.time_dim == 0\
                else geometry_vector[self.FULL_STOP_INDEX, :]
        full_stop_point_index = np.where(full_stop_slice == 1.0)[0]

        if len(full_stop_point_index) <= 0:
            # we lack an end point (trimmed?)
            full_stop_point_index = geometry_vector.shape[self.time_dim]
        else:
            full_stop_point_index = full_stop_point_index[0]

        if full_stop_point_index == 0:
            # we're a point
            full_stop_point_index = 1

        return full_stop_point_index

    def localized_mean(self, geometry_vector):
        full_stop_point_index = self.get_full_stop_index(geometry_vector)
        geom_mean = geometry_vector[:full_stop_point_index, 2:4].mean(axis=0)\
            if self.time_dim == 0\
            else geometry_vector[2:4, :full_stop_point_index].mean(axis=1)

        return geom_mean
