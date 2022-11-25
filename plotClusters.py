#!/usr/bin/env python

from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Please install 'matplotlib' via pip")
import numpy as np


def make_plot(data):
    # compute optimal grid layout heuristically
    num_plots = len(data)
    plot_dims = np.array([[i, j, abs(i-j)]
                          for i in range(1, num_plots+1)
                          for j in range(1, num_plots+1)
                          if i*j == num_plots])
    plot_dims = plot_dims[plot_dims[:,2] == plot_dims[:,2].min()][0]
    rows, cols = plot_dims[0], plot_dims[1]

    if len(data) >= 3 and (rows == 1 or cols == 1):
        plot_dims = np.array([[i, j, abs(i-j)]
                              for i in range(1, num_plots+1)
                              for j in range(1, num_plots+1)
                              if i+j == num_plots])


        plot_dims = plot_dims[plot_dims[:,2] == plot_dims[:,2].min()][0]
        rows = max(plot_dims[0], plot_dims[1])
        cols = max(plot_dims[0], plot_dims[1])

    figure, axis = plt.subplots(rows, cols, sharex=True, sharey=True)
    if not isinstance(axis, np.ndarray):
        # only one plot
        axis = np.array(axis)
        axis = np.expand_dims(axis, axis=(0, 1))

    if axis.ndim <= 1:
        # exactly two plots
        axis = np.expand_dims(axis, axis=0)

    i = 0
    for k, nepoch in enumerate(sorted(data.keys())):
        clusters = np.loadtxt(data[nepoch])

        j = k%cols
        axis[i, j].scatter(clusters[:, 0],
                           clusters[:, 1],
                           marker='+',
                           s=2,
                           c='#000000')
        axis[i, j].set_title(f"{nepoch} epoch")

        if (j + 1) >= cols:
            i += 1

    figure.suptitle("Entity Embeddings Space")
    plt.tight_layout()

    return plt

def main(work_dir):
    work_dir = Path(work_dir)

    data = dict()
    for file in work_dir.glob('node_embedding_clusters_*_epoch.gz'):
        nepoch = file.name.split('_')[3]
        try:
            nepoch = int(nepoch)
        except:
            continue

        data[nepoch] = file

    return make_plot(data)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Run this as `python plotClusters.py <work dir>'"
    work_dir = sys.argv[1]

    plots = main(work_dir)
    plots.show()
