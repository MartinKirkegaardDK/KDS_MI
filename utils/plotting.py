from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch
from sklearn.decomposition import PCA

def plot_activations_PCA(
        activations_by_language: dict[str, list[torch.Tensor]],
        ax: Axes,
        layer: int
    ) -> None:
    '''
    Takes lists of activation and a matplotlib ax, and plots a scatterplot of a PCA of them.

    Args:
        activations_by_language: dictionary with keys as langauges and values as list of activation tensors
        ax: the axes to plot them on
    '''

    # fits PCA using the *average* of each language 
    # -- ensures that the principle components are the ones relevant to *distinguishing* the languages from each other
    avg_vectors = torch.stack(
        tensors=[
            torch.mean(torch.stack(activations), dim=0)
            for activations in activations_by_language.values()
        ],
        dim=0
    )
    pca = PCA(n_components=2).fit(avg_vectors.cpu())


    # transforms all vectors according to the fitted PCA
    languages = []
    for language, activations in activations_by_language.items():
        languages.extend([language] * len(activations))

    vectors = torch.concat(
        tensors=[
            torch.stack(activations, dim=0)
            for activations in activations_by_language.values()
        ],
        dim=0
    )
    transformed = pca.transform(vectors.cpu())


    # creates a mapping from each language name to an index 
    # -- necessary because of the required format for ax.scatter()
    mapping = {language: idx for idx, language in enumerate(activations_by_language.keys())}
    reverse_mapping = {idx: language for language, idx in mapping.items()}

    
    # plots the transformed activations
    x, y = transformed.T 
    scatter = ax.scatter(x, y, c=list(map(lambda x: mapping[x], languages)), alpha=0.4, s=2)
    ax.legend(handles=scatter.legend_elements()[0], labels=[reverse_mapping[i] for i in range(len(reverse_mapping))])
    ax.set_title(f'PCA of layer {layer}')
