import torch
from matplotlib import pyplot as plt
from typing import Callable

from utils.plotting import plot_PCA
from math import sqrt

def run(steering_vector_paths_by_language: dict[str, Callable], hidden_layers: int):


    # initializes the figure with the correct amount of rows and columns
    ncols = int(sqrt(hidden_layers)) + int((hidden_layers % hidden_layers) != 0)
    nrows = hidden_layers // ncols + int(hidden_layers % ncols != 0)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,20))
    axs = axs.flatten()

    for layer in range(hidden_layers):

        # loads steering vectors for eacg language
        steering_vectors_by_language = {
            language: torch.load(steering_vector_paths_by_language[language](layer), map_location='cpu', weights_only=True)
            for language in steering_vector_paths_by_language.keys()
        }

        # plots steering vectors on a PCA plot
        plot_PCA(
            steering_vectors=steering_vectors_by_language, 
            layer=layer, 
            ax=axs[layer]
        )

    fig.tight_layout()
    
