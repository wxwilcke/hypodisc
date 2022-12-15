#! /usr/bin/env python

from hypodisc.models import DenoisingAE


def reduce_dimensions(embeddings, target_dimensions=2, num_layers=3):
    """
    Reduce dimensions of 2D embedding tensor by stacking denoising encoders.
    """
    step_size = (embeddings.shape[1]-target_dimensions)//num_layers
    dimensions = [(target_dimensions + ((i + 1) * step_size), 
                   target_dimensions + (i * step_size))
                  for i in reversed(range(num_layers))]

    for layer, (input_dim, bneck_dim) in enumerate(dimensions):
        denoisingAE = DenoisingAE(input_dim=input_dim,
                                  bneck_dim=bneck_dim)


