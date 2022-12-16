#! /usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from hypodisc.models import DenoisingAE


def reduce_dimensions(data, target_dimensions=2, num_layers=3):
    """
    Reduce dimensions of 2D embedding tensor by stacking denoising encoders.
    """
    # TODO: make these flags
    num_epoch = 50
    batchsize = 1024


    step_size = (data.shape[1]-target_dimensions)//num_layers
    dimensions = [(target_dimensions + ((i + 1) * step_size), 
                   target_dimensions + (i * step_size))
                  for i in reversed(range(num_layers))]


    for layer_id, (input_dim, bneck_dim) in enumerate(dimensions):
        mode_str = "[RDDIM] %3.d " % layer_id
        print(mode_str, end='', flush=True)

        model = DenoisingAE(input_dim=input_dim,
                            bneck_dim=bneck_dim,
                            real_valued=True)
        optimizer = optim.Adam(model.parameters())
        loss = nn.MSELoss()

        model.train()
        for epoch in range(num_epoch):
            # prepare batches
            # TODO: randomly order per epoch
            num_samples = data.shape[0]
            batches = [slice(begin, min(begin+batchsize, num_samples))
                       for begin in range(0, num_samples, batchsize)]
            num_batches = len(batches)

            loss_lst = list()
            for batch_id, batch in enumerate(batches):
                batch_str = " - epoch %3.d - batch %2.d / %d" % (epoch,
                                                                 batch_id+1,
                                                                 num_batches)
                print(batch_str, end='\b'*len(batch_str), flush=True)

                batch_data = data[batch.start:batch.stop]
                batch_out = model(batch_data)

                # MSE loss between original and reconstructed data
                batch_loss = loss(batch_data, batch_out)
            
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                batch_loss.backward()  # training loss
                optimizer.step()

                batch_loss = float(batch_loss)
                loss_lst.append(batch_loss)

            print(np.mean(loss_lst))
