#! /usr/bin/env python

import h5py
import numpy as np


UNT1DFIXL = 0
UNT1DVARL = 1
UNT2DFIXL = 2
UNT2DVARL = 3
UNTNDFIXL = 4
UNTNDVARL = 5
INT1DFIXL = 6
INT1DVARL = 7
INT2DFIXL = 8
INT2DVARL = 9
INTNDFIXL = 10
INTNDVARL = 11
FLT1DFIXL = 12
FLT1DVARL = 13
FLT2DFIXL = 14
FLT2DVARL = 15
FLTNDFIXL = 16
FLTNDVARL = 17

_MODALITIES = ["textual", "numerical", "temporal", "visual", "spatial"]


class HDF5:
    _hf = None
    _compression = None

    def __init__(self, path, mode='r', compression="lzf"):
        self._hf = h5py.File(path, mode)
        self._compression = compression if compression != "none" else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._hf.close()

    def read_dataset(self, modalities=None):
        dataset = dict()

        dataset['num_nodes'] = self._read_metadata(self._hf,
                                                   'num_nodes')
        dataset['num_relations'] = self._read_metadata(self._hf,
                                                       'num_relations')

        group = self._hf["data"]
        dataset |= self._read_data(group)

        modalities = modalities if modalities is not None else _MODALITIES
        for modality in modalities:
            if modality not in self._hf.keys():
                continue

            data = list()
            group = self._hf[modality]
            for datatype in group.keys():
                data.append(self._read_data_from_group(group, datatype))

            dataset[modality] = data

        return dataset

    def write_dataset(self, dataset):
        self.write_metadata('num_nodes', dataset['num_nodes'])
        self.write_metadata('num_relations', dataset['num_relations'])
        self.write_data(dataset)
        for modality in _MODALITIES:
            if modality in dataset.keys():
                self.write_modality_data(dataset[modality], modality)

    def write_metadata(self, key, value):
        self._write_metadata(self._hf, key, value)

    def write_modality_data(self, data, modality):
        group = self._hf[modality] if modality in self._hf.keys()\
                else self._hf.create_group(modality)
        for datatype_set in data:
            self._write_data_to_group(group, datatype_set,
                                      self._compression)

    def write_data(self, dataset):
        group = self._hf.create_group("data")
        self._write_data(group, dataset, compression=self._compression)

    def _write_metadata(self, group, name, data):
        group.attrs[name] = data

    def _write_data(self, group, dataset, compression):
        triples = dataset['triples']
        entities = dataset['entities']
        train = dataset['trainset']
        test = dataset['testset']
        valid = dataset['validset']

        group.create_dataset("trainset", data=train,
                             compression=compression)
        group.create_dataset("testset", data=test,
                             compression=compression)
        group.create_dataset("validset", data=valid,
                             compression=compression)

        group.create_dataset("triples", data=triples,
                             compression=compression)
        group.create_dataset("entities", data=entities,
                             compression=compression)

    def _write_data_to_group(self, group, dtype_data, compression):
        datatype, data, data_length, data_entity_map,\
                is_varlength, time_dim, hdftype = dtype_data
        xsdtype = datatype.split('/')[-1]
        subgroup = group.create_group(xsdtype)

        dtype = 'f4'
        if hdftype <= UNTNDVARL:
            dtype = 'u1'
        elif hdftype <= INTNDVARL:
            dtype = 'i4'

        # stack and write data
        if hdftype in [UNT1DVARL, INT1DVARL]:
            self._write_var_length_1D(subgroup, "data",
                                      data, data_length,
                                      dtype,
                                      compression=compression)
        elif hdftype in [UNT2DFIXL, FLT2DFIXL]:
            self._write_fixed_length_2D(subgroup, "data",
                                        data, data_length,
                                        dtype,
                                        compression=compression)
        elif hdftype in [UNT2DVARL, INT2DVARL, FLT2DVARL]:
            self._write_var_length_2D(subgroup, "data",
                                      data, data_length,
                                      time_dim, dtype,
                                      compression=compression)
        elif hdftype in [UNTNDFIXL, FLTNDFIXL]:
            self._write_fixed_length_ND(subgroup, "data",
                                        data, data_length,
                                        dtype,
                                        compression=compression)
        else:
            raise NotImplementedError(hdftype)

        # write lengths and mapping index
        subgroup.create_dataset("data_length",
                                data=np.array(data_length),
                                compression=compression)
        subgroup.create_dataset("data_entity_map",
                                data=np.array(data_entity_map),
                                compression=compression)

        # write metada
        subgroup.attrs['is_varlength'] = is_varlength
        subgroup.attrs['time_dim'] = time_dim
        subgroup.attrs['hdftype'] = hdftype

    def _write_fixed_length_2D(self, subgroup, name,
                               data, data_length,
                               dtype, compression):
        nrows = len(data)
        ncols = max(data_length)

        f = subgroup.create_dataset(name, shape=(nrows, ncols),
                                    dtype=dtype, compression=compression)
        for i in range(nrows):
            # items are lists or raw values
            row = data[i]
            f[i] = np.array(row)

    def _write_fixed_length_ND(self, subgroup, name,
                               data, data_length,
                               dtype, compression):
        shapes = set(item.shape for item in data)
        assert len(shapes) == 1, "Shapes of variable size not supported"
        shape = shapes.pop()

        num_items = len(data)
        f = subgroup.create_dataset(name, shape=(num_items, *shape),
                                    dtype=dtype, compression=compression)
        for i in range(num_items):
            # items are ndarrays
            f[i] = data[i]

    def _write_var_length_1D(self, subgroup, name,
                             data, data_length,
                             dtype, compression):
        # assumes tokens
        # stack on time_dim
        full_length = sum(data_length)

        offset = 0
        num_items = len(data)
        shape = (full_length,)
        f = subgroup.create_dataset(name, shape=shape, dtype=dtype,
                                    compression=compression)
        for i in range(num_items):
            # items are ndarrays
            item = data[i]
            item_length = data_length[i]
            f[offset:offset+item_length] = item

            offset += item_length

    def _write_var_length_2D(self, subgroup, name,
                             data, data_length, time_dim,
                             dtype, compression):
        # stack on time_dim
        alphabet_size = set(item.shape[1-time_dim] for item in data)
        assert len(alphabet_size) == 1, "Alphabets of variable size not "\
                                        "supported"

        alphabet_size = alphabet_size.pop()
        full_length = sum(data_length)

        offset = 0
        num_items = len(data)
        if time_dim == 0:
            shape = (full_length, alphabet_size)
            f = subgroup.create_dataset(name, shape=shape, dtype=dtype,
                                        compression=compression)
            for i in range(num_items):
                # items are ndarrays
                item = data[i]
                item_length = item.shape[time_dim]
                f[offset:offset+item_length, :] = item

                offset += item_length
        elif time_dim == 1:
            shape = (alphabet_size, full_length)
            f = subgroup.create_dataset(name, shape=shape, dtype=dtype,
                                        compression=compression)
            for i in range(num_items):
                item = data[i]
                item_length = item.shape[time_dim]
                f[:, offset:offset+item_length] = item

                offset += item_length
        else:
            raise Exception("Unsupported time dimension %d" % time_dim)

    def _read_metadata(self, group, name):
        return group.attrs[name]

    def _read_data(self, group):
        out = dict()

        for name in ["triples", "entities", "trainset", "testset", "validset"]:
            out[name] = np.array(group.get(name))

        return out

    def _read_data_from_group(self, group, datatype):
        subgroup = group[datatype]

        # read metadata
        is_varlength = subgroup.attrs['is_varlength']
        time_dim = subgroup.attrs['time_dim']
        hdftype = subgroup.attrs['hdftype']

        dtype = np.float32
        if hdftype <= UNTNDVARL:
            dtype = np.uint8
        elif hdftype <= INTNDVARL:
            dtype = np.int32

        # read length and index mapping
        data_length = np.array(subgroup.get("data_length"))
        data_entity_map = np.array(subgroup.get("data_entity_map"))

        # read data
        data = None
        if hdftype in [UNT2DVARL, UNTNDVARL, INT2DVARL, INTNDVARL,
                       FLT2DVARL, FLTNDVARL]:
            data = self._read_var_length_ND(subgroup, "data", dtype,
                                            data_length, time_dim)
        elif hdftype in [UNT2DFIXL, UNTNDFIXL, INT2DFIXL, INTNDFIXL,
                         FLT2DFIXL, FLTNDFIXL]:
            data = self._read_fixed_length_ND(subgroup, "data", dtype)
        elif hdftype in [UNT1DVARL, INT1DVARL, FLT1DVARL]:
            data = self._read_var_length_1D(subgroup, "data", dtype,
                                            data_length)
        else:
            raise Exception()

        return (datatype, data, data_length, data_entity_map,
                is_varlength, time_dim)

    def _read_fixed_length_ND(self, subgroup, name, dtype):
        out = list()
        data = np.array(subgroup.get(name), dtype=dtype)
        for i in range(data.shape[0]):
            item = data[i]

            if item.ndim == 1 and len(item) == 1:
                item = [item[0]]

            out.append(item)

        return out

    def _read_var_length_1D(self, subgroup, name, dtype, data_length):
        out = list()
        offset = 0
        data = np.array(subgroup.get(name), dtype=dtype)
        for item_length in data_length:
            item = data[offset:offset+item_length]

            offset += item_length
            out.append(item)

        return out

    def _read_var_length_ND(self, subgroup, name, dtype,
                            data_length, time_dim):
        out = list()
        offset = 0
        data = np.array(subgroup.get(name), dtype=dtype)
        if time_dim == 0:
            for item_length in data_length:
                item = data[offset:offset+item_length, :]

                offset += item_length
                out.append(item)
        elif time_dim == 1:
            for item_length in data_length:
                item = data[:, offset:offset+item_length]

                offset += item_length
                out.append(item)
        else:
            raise Exception("Unsupported time dimension %d" % time_dim)

        return out
