#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypodisc.utils import getConfParam, zero_pad
from hypodisc.vectorizers.visual import normalize as im_norm


_DIM_DEFAULT = {"numerical": 4,
                "temporal": 16,
                "textual": 128,
                "spatial": 128,
                "visual": 128}

def loadFromHub(config):
    parameters = list()
    named_parameters = dict()

    for param in config:
        if param is None or '=' not in param:
            parameters.append(param)
            continue

        key, value = param.split('=')
        named_parameters[key.strip()] = value.strip()

    return torch.hub.load(*parameters, **named_parameters)

def freeze_(model, layer='', _grad=False):
    """ Freeze one or more layers
    """
    for name, param in model.named_parameters():
        if layer in name:  # '' matches all
            param.requires_grad_(_grad)

def unfreeze_(model, layer=''):
    freeze_(model, layer, _grad=True)


""" Modules """


class NeuralEncoders(nn.Module):
    def __init__(self,
                 dataset,
                 config):
        """
        Neural Encoder(s)

        """
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.modalities = dict()
        self.positions = dict()
        self.out_dim = 0

        language_model = None
        image_model = None

        pos = 0
        for modality in dataset.keys():
            if len(dataset[modality]) <= 0:
                continue

            if modality not in self.modalities.keys():
                self.modalities[modality] = list()

            inter_dim = getConfParam(config,
                                     '.'.join([modality, "output_dim"]), 
                                     _DIM_DEFAULT[modality])
            dropout = getConfParam(config,
                                   '.'.join([modality, "dropout"]),
                                   0.0)                                   
            bias = getConfParam(config,
                                '.'.join([modality, "bias"]),
                                False)
            weights = getConfParam(config,
                                   '.'.join([modality, "weights"]),
                                   None)

            encoder = None
            for mset in dataset[modality]:
                datatype = mset[0].split('/')[-1]
                seq_lengths = -1

                if modality == "numerical":
                    encoder = MLP(input_dim=1, output_dim=inter_dim,
                                  num_layers=1,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "textual":
                    if language_model is None:
                        hub_call = ["huggingface/pytorch-transformers",
                                    "model",
                                    weights]
                        language_model = loadFromHub(hub_call)

                    encoder = Transformer(output_dim=inter_dim,
                                          base_model=language_model,
                                          p_dropout=dropout,
                                          bias=bias)
                elif modality == "temporal":
                    f_in = mset[1][0].shape[0]
                    encoder = MLP(input_dim=f_in, output_dim=inter_dim,
                                  num_layers=1,
                                 p_dropout=dropout,
                                 bias=bias)
                elif modality == "visual":
                    if image_model is None:
                        hub_call = ["pytorch/vision:v0.10.0",
                                    "mobilenet_v2",
                                    weights]
                        image_model = loadFromHub(hub_call)
                        image_model = image_model.features  # omit classifier

                    encoder = ImageCNN(features_out=inter_dim,
                                       base_model=image_model,
                                       p_dropout=dropout,
                                       bias=bias)
                elif modality == "spatial":
                    time_dim = mset[-1]
                    f_in = mset[1][0].shape[1-time_dim]  # vocab size

                    seq_lengths = mset[2]
                    seq_length_q25 = np.quantile(seq_lengths, 0.25)
                    if seq_length_q25 < TCNN.LENGTH_M:
                        seq_lengths = TCNN.LENGTH_S
                    elif seq_length_q25 < TCNN.LENGTH_L:
                        seq_lengths = TCNN.LENGTH_M
                    else:
                        seq_lengths = TCNN.LENGTH_L

                    encoder = TCNN(features_in=f_in,
                                   features_out=inter_dim,
                                   p_dropout=dropout,
                                   size=seq_lengths,
                                   bias=bias)

                self.encoders[datatype] = encoder
                self.modalities[modality].append(encoder)
                self.out_dim += inter_dim

                pos_new = pos + inter_dim
                self.positions[datatype] = (pos, pos_new)
                pos = pos_new

    def forward(self, features):
        # batch_idx := global indices of nodes to compute embeddings for
        data, batch_idx, device = features

        batchsize = len(batch_idx)
        batch_out_dev = torch.zeros((batchsize, self.out_dim),
                                    dtype=torch.float32, device=device)
        for msets in data.values():
            for mset in msets:
                datatype, X, _, X_idx, _, time_dim = mset
                datatype = datatype.split('/')[-1]
                if datatype not in self.encoders.keys():
                    continue

                encoder = self.encoders[datatype]
                pos_begin, pos_end = self.positions[datatype]

                # filter nodes that do not have this datatype
                # same as intersection, but ensures order
                batch_idx_local = [i for i in range(len(batch_idx))
                                   if batch_idx[i] in X_idx]
                batch_idx_filtered = batch_idx[batch_idx_local]

                # skip if no nodes in this batch have this datatype
                if len(batch_idx_filtered) <= 0:
                    continue

                # gather relevant row numbers of X (ordered)
                X_idx_inv = {e_idx: i for i,e_idx in enumerate(X_idx)}
                X_batch_idx = [X_idx_inv[e_idx]
                               for e_idx in batch_idx_filtered]

                # build batch subset of X with relevant node values
                if datatype in ["XMLSchema#string", "XMLSchema#anyURI"]:
                    X = [torch.tensor(X[i], dtype=torch.int32)
                         for i in X_batch_idx]
                else:
                    X = [torch.tensor(X[i], dtype=torch.float32)
                         for i in X_batch_idx]

                # normalize images if present
                if datatype == "dt#base64Image":
                    X = [im_norm(x) for x in X]

                # stack individual tensors and pad if different lengths
                X_batch = torch.stack(zero_pad(X, time_dim), dim=0)
                if isinstance(encoder, Transformer):
                    X_batch.squeeze_()

                X_batch_dev = X_batch.to(device)

                # compute output
                out_dev = encoder(X_batch_dev)

                # map output to correct position on Y
                batch_out_dev[batch_idx_local, pos_begin:pos_end] = out_dev

        return batch_out_dev


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=3,
                 p_dropout=0.0,
                 bias=False):
        """
        Multi-Layer Perceptron with N layers

        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_dropout = p_dropout
        step_size = (input_dim-output_dim)//num_layers
        hidden_dims = [output_dim + (i * step_size)
                       for i in reversed(range(num_layers))]

        mlp = list()
        layer_indim = input_dim
        for hidden_dim in hidden_dims:
            mlp.extend([nn.Linear(layer_indim, hidden_dim, bias),
                        nn.Dropout(p=self.p_dropout, inplace=True),
                        nn.PReLU()])

            layer_indim = hidden_dim

        self.mlp = nn.Sequential(*mlp)

        # initiate weights
        self.init()

    def forward(self, X):
        return self.mlp(X)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)


class DenoisingAE(nn.Module):
    NOISE_GAUSSIAN = 0
    NOISE_MASKING = 1
    NOISE_SALTANDPEPPER = 2

    NOISE_GAUSSIAN_ALPHA = 0.1
    NOISE_MASKING_PROB = 0.1
    NOISE_SALTANDPEPPER_PROB = 0.1

    def __init__(self,
                 input_dim,
                 bneck_dim,
                 noise=NOISE_SALTANDPEPPER,
                 modifier=None,
                 tied_weights=False,
                 bias=False):
        """
        Denoising Autoencoder

        :param input_dim: original dimensions of input (and output) vector
        :param bneck_dim: dimensions of bottleneck (learned representation)
        :param noise: type of noise to apply to input vector
        :param modifier: noise-specific modifier parameter
        :param tied_weights: whether to tie the weight matrices
        :paran bias: whether to add bias nodes to the layers


        As introduced in:

        Vincent, P. et al.: Stacked denoising autoencoders: Learning useful
        representations in a deep network with a local denoising criterion.
        Journal of machine learning research. 11, 12, (2010).
        """
        super().__init__()

        assert noise in [0,1,2], "Unknown type of noise provided"
        self.noise = noise

        self.modifier = modifier
        if self.modifier is None:
            if self.noise == self.NOISE_GAUSSIAN:
                self.modifier = self.NOISE_GAUSSIAN_ALPHA
            elif self.noise == self.NOISE_MASKING:
                self.modifier = self.NOISE_MASKING_PROB
            elif self.noise == self.NOISE_SALTANDPEPPER:
                self.modifier = self.NOISE_SALTANDPEPPER_PROB

        self.encoder = self.Encoder(input_dim = input_dim,
                                    bneck_dim = bneck_dim,
                                    bias = bias)
        if tied_weights:
            self.decoder = self.Decoder(output_dim = input_dim,
                                        bneck_dim = bneck_dim,
                                        bias = bias,
                                        tie_weights_with = self.encoder.W)
        else:
            self.decoder = self.Decoder(output_dim = input_dim,
                                        bneck_dim = bneck_dim,
                                        bias = bias)

        # initiate weights
        self.init()

    def forward(self, X):
        # corrupt copy of original input data
        X = self.corrupt(X.clone(), self.noise, self.modifier)

        Y = self.encoder(X)
        Z = self.decoder(Y)

        return Z

    def corrupt(self, X, noise, modifier):
        if noise == self.NOISE_GAUSSIAN:
            X += modifier * torch.normal(mean=X.mean(),
                                         std=X.std(),
                                         shape=X.shape)
        elif noise == self.NOISE_MASKING:
            mask = torch.bernoulli(torch.tensor([modifier] * X.numel()))
            mask = mask.type(torch.bool).view(X.shape)
            X[mask] = 0.
        elif noise == self.NOISE_SALTANDPEPPER:
            mask = torch.bernoulli(torch.tensor([modifier] * X.numel()))
            mask = mask.type(torch.bool).view(X.shape)
            X[mask] = torch.bernoulli(torch.tensor([.5] * mask.sum()))

        return X

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)


    class Encoder(nn.Module):
        def __init__(self,
                     input_dim,
                     bneck_dim,
                     bias=False):

            super().__init__()
        
            # weight matrix
            self.W = nn.Parameter(torch.empty((input_dim, bneck_dim)))

            self.b = None
            if bias:
                self.b = nn.Parameter(torch.empty(bneck_dim))
        
            # NB: original paper uses sigmoid
            self.activation = nn.ReLU(inplace=True)

        def forward(self, X):
            Y = torch.matmul(X, self.W)
            if self.b is not None:
                Y += self.b

            return self.activation(Y)  # bottleneck

        def init(self):
            for param in self.parameters():
                nn.init.uniform_(param)


    class Decoder(nn.Module):
        def __init__(self,
                     bneck_dim,
                     output_dim,
                     bias=False,
                     tie_weights_with=None):

            super().__init__()
        
            # weight matrix
            self.tied_weights = False
            if tie_weights_with is not None:
                self.W = tie_weights_with  # encoder weight matrix

                self.tied_weights = True
            else:
                self.W = nn.Parameter(torch.empty((bneck_dim, output_dim)))

            self.b = None
            if bias:
                self.b = nn.Parameter(torch.empty(output_dim))
        
            # NB: original paper uses sigmoid
            self.activation = nn.ReLU(inplace=True)

        def forward(self, Y):
            if self.tied_weights:
                Z = torch.einsum('ij,kj->ik', Y, self.W)  # Y x Wenc^T
            else:
                Z = torch.matmul(Y, self.W)

            if self.b is not None:
                Z += self.b

            return self.activation(Z)  # reconstructed X

        def init(self):
            for param in self.parameters():
                nn.init.uniform_(param)



class ImageCNN(nn.Module):
    def __init__(self, features_out, base_model, p_dropout=0.2,
                 bias=True, finetune=True):
        super().__init__()

        self.module_dict = nn.ModuleDict()

        self.base_model = base_model
        self.finetune = finetune
        if self.finetune:
            freeze_(self.base_model)
        self.module_dict['pretrained_head'] = self.base_model
        
        inter_dim = self.base_model[-1].out_channels
        self.pre_fc = nn.Linear(inter_dim, inter_dim, bias=bias)
        self.fc = nn.Linear(inter_dim, features_out, bias=bias)
        self.f_activation = nn.PReLU()
        self.module_dict['pre_fc'] = self.pre_fc
        self.module_dict['fc'] = self.fc
        self.module_dict['activation'] = self.f_activation

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.module_dict['pool'] = self.avgpool
        self.dropout = None
        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
            self.module_dict['dropout'] = self.dropout

    def forward(self, X):
        output = self.base_model(X)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)

        output = self.pre_fc(output)
        output = self.f_activation(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return self.fc(output)

class Transformer(nn.Module):
    def __init__(self, output_dim, base_model, p_dropout=0.2,
                 bias=True, finetune=True):
        super().__init__()

        self.module_dict = nn.ModuleDict()

        self.base_model = base_model
        self.finetune = finetune
        if self.finetune:
            freeze_(self.base_model)
        self.module_dict['pretrained_head'] = self.base_model
        
        inter_dim = list(self.base_model.modules())[-1].normalized_shape[0]
        self.pre_fc = nn.Linear(inter_dim, inter_dim, bias=bias)
        self.fc = nn.Linear(inter_dim, output_dim, bias=bias)
        self.f_activation = nn.PReLU()
        self.module_dict['pre_fc'] = self.pre_fc
        self.module_dict['fc'] = self.fc
        self.module_dict['activation'] = self.f_activation

        self.dropout = None
        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
            self.module_dict['dropout'] = self.dropout

    def forward(self, X):
        hidden_state = self.base_model(X.long())[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_fc(pooled_output)
        pooled_output = self.f_activation(pooled_output)

        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)

        return self.fc(pooled_output)

class TCNN(nn.Module):
    LENGTH_S = 20
    LENGTH_M = 100
    LENGTH_L = 300

    def __init__(self, features_in, features_out, p_dropout=0.0, bias=True,
                 size="M"):
        """
        Temporal Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        """
        super().__init__()

        if size == self.LENGTH_S:
            self.minimal_length = self.LENGTH_S
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(2),

                nn.Conv1d(256, 512, kernel_size=2, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )

            n_first = max(256, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(512, n_first, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.PReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )
        elif size == self.LENGTH_M:
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 1024, kernel_size=3, padding=0),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True)
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(1024, n_first, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.PReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )
        elif size == self.LENGTH_L:
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Conv1d(1024, 2048, kernel_size=3, padding=0),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True)
            )

            n_first = max(512, features_out)
            n_second = max(128, features_out)
            self.fc = nn.Sequential(
                nn.Linear(2048, n_first, bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_first, n_second, bias),
                nn.PReLU(),
                nn.Dropout(p=p_dropout),

                nn.Linear(n_second, features_out, bias)
            )

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X


class DistMult(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_relations,
                 embedding_dim=-1,
                 literalE=False,
                 **kwargs):
        """
        """
        super().__init__()

        if embedding_dim < 0:
            embedding_dim = num_nodes

        # matrix of node embeddings
        self.node_embeddings = nn.Parameter(torch.empty((num_nodes,
                                                         embedding_dim)))
        # simulate diag(R) by vectors (r x h)
        self.edge_embeddings = nn.Parameter(torch.empty((num_relations,
                                                         embedding_dim)))

        self.fuse_model = None
        if literalE:
            # feature_dim must be given by kwargs
            self.fuse_model = LiteralE(embedding_dim=embedding_dim,
                                       **kwargs)

        # initiate weights
        self.reset_parameters()

    def forward(self, X):
        # data := rows of triples by global indices;
        #         indices must map to local embedding tensors
        # E_idx := global indices of entities by order of embeddings
        # feature_embeddings := literal embeddings belonging to entities
        (e_idc, p_idc, u_idc), (E_batch_idx, feature_embeddings) = X

        p = self.edge_embeddings[p_idc, :]
        e = self.node_embeddings[e_idc, :]
        u = self.node_embeddings[u_idc, :]
        if self.fuse_model is not None:
            # fuse entity and literal embeddings
            embeddings = self.fuse_model([self.node_embeddings[E_batch_idx],
                                          feature_embeddings])

            # map global index to row of corresponding embeddings
            E_batch_idx_inv = {e_idx.item(): i
                               for i, e_idx in enumerate(E_batch_idx)}

            # replace default embeddings of left-hand side by fused ones
            e_mask = torch.isin(e_idc, E_batch_idx)
            e_emb_idx = [E_batch_idx_inv[e_idx.item()]
                         for e_idx in e_idc[e_mask]]
            e[e_mask, :] = embeddings[e_emb_idx, :]

            # replace default embeddings of right-hand side by fused ones
            u_mask = torch.isin(u_idc, E_batch_idx)
            u_emb_idx = [E_batch_idx_inv[u_idx.item()]
                         for u_idx in u_idc[u_mask]]
            u[u_mask, :] = embeddings[u_emb_idx, :]

        # optimizations for common broadcasting
        if len(e.size()) == len(p.size()) == len(u.size()):
            if p_idc.size(-1) == 1 and u_idc.size(-1) == 1:
                singles = p * u
                return torch.matmul(e, singles.transpose(-1, -2)).squeeze(-1)

            if e_idc.size(-1) == 1 and u_idc.size(-1) == 1:
                singles = e * u
                return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

            if e_idc.size(-1) == 1 and p_idc.size(-1) == 1:
                singles = e * p
                return torch.matmul(u, singles.transpose(-1, -2)).squeeze(-1)

        return torch.sum(e * p * u, dim=-1)

    def reset_parameters(self, one_hot=False):
        for name, param in self.named_parameters():
            if name in ["node_embeddings", "edge_embeddings"]:
                if one_hot:
                    nn.init.eye_(param)
                else:
                    nn.init.normal_(param)


class LiteralE(nn.Module):
    def __init__(self,
                 embedding_dim,
                 feature_dim):
        """
        LiteralE embedding model

        embedding_dim :: length of entity vector (H in paper)
        feature_dim  :: length of entity feature vector (N_d in paper)

        NB: Different from LiteralE, this implementation lets feature matrix
            L = N_e x F, where F is a concatenation of the outputs of all
            relevant encoders.
        """
        super().__init__()

        self.W_ze = nn.Parameter(torch.empty((embedding_dim,
                                              embedding_dim)))
        self.W_zl = nn.Parameter(torch.empty((feature_dim,
                                              embedding_dim)))

        # split W_h in W_he and W_hl for cheaper computation
        self.W_he = nn.Parameter(torch.empty((embedding_dim,
                                              embedding_dim)))
        self.W_hl = nn.Parameter(torch.empty((feature_dim,
                                              embedding_dim)))

        self.b = nn.Parameter(torch.empty((embedding_dim)))

        # initiate weights
        self.reset_parameters()

    def forward(self, X):
        # E := length H
        # L := length F (N_d in paper)
        E, L = X

        Wze = torch.einsum('ij,ki->kj', self.W_ze, E)
        Wzl = torch.einsum('ij,ki->kj', self.W_zl, L)

        Z = torch.sigmoid(Wze + Wzl + self.b)  # out H
        del Wze, Wzl

        Whe = torch.einsum('ij,ki->kj', self.W_he, E)
        Whl = torch.einsum('ij,ki->kj', self.W_hl, L)

        H = torch.tanh(Whe + Whl)  # out H
        del Whe, Whl

        # compute result of function g
        return Z * H + (1 - Z) * E

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param)


""" Loss functions """


class BCEWithLogitsAndClusterLoss(nn.Module):
    """Binary cross-entropy and cluster loss."""
    __name__ = 'BCEWithLogitsAndClusterLoss'

    # TODO: add function to adapt value of a to number of epoch or loss
    # similar to ADAM (first a=1, then gradually reduce to a=0.5)

    def __init__(self, a = 0.5, **kwargs):
        """
        param   a: trade off between BCE and cluster loss
        """
        super().__init__()

        assert 0. <= a <= 1., "parameter value must be in range [0,1]"

        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss(**kwargs)
        self.cluster_loss = self._cluster_loss
        self.a = a

    def forward(self, inputs, targets, embeddings, a = None):
        # use global or local parameter
        a = self.a if a is None else a
        assert a >= 0. and a <= 1., "parameter value must be in range [0,1]"

        bce_loss, cls_loss = 0., 0.
        if a > 0.:
            bce_loss = self.binary_cross_entropy_loss(inputs, targets)
        if a < 1.:
            cls_loss = self.cluster_loss(embeddings)

        return a * bce_loss + (1 - a) * cls_loss

    def _cluster_loss(self, embeddings):
        """
        Density-based clustering loss
        """
        return 0.  # TODO
