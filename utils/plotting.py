from collections import defaultdict
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import sem, t
from sklearn.decomposition import PCA

def plot_activations_PCA(
        activations_by_language: dict[str, list[torch.Tensor]],
        layer: int,
        ax: Axes,
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



def plot_probe_results(
        accuracies_by_layer: dict[int, list[dict[int, float]]],
        reg_lambda: float,
        map_lab: dict[int, str],
        ax: Axes,
    ) -> None:
    '''
    Takes lists of bootstrapped probe accuracies for each layer and plots them on the given matplotlib axes

    Args:
        accuracies_by_layer: dictionary with layers as keys and lists of bootstrapped accuracies for each label as values: {layer: [{label: accuracy}]}
    '''

    label_stats = defaultdict(lambda: defaultdict(list))
    
    # Collect all values per label per layer
    for layer, samples in accuracies_by_layer.items():
        for sample in samples:
            for label, value in sample.items():
                label_stats[label][int(layer)].append(value)
    
    
    # Use a distinct color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Slightly offset x positions for different labels to avoid direct overlap
    offset_step = 0.1
    
    for idx, (label, layer_values) in enumerate(label_stats.items()):
        layers = sorted(layer_values.keys())
        means = [np.mean(layer_values[layer]) for layer in layers]
        conf_intervals = [sem(layer_values[layer]) * t.ppf((1 + 0.95) / 2, len(layer_values[layer]) - 1) for layer in layers]
        
        # Create upper and lower bounds for the confidence interval
        upper_bound = [means[i] + conf_intervals[i] for i in range(len(means))]
        lower_bound = [means[i] - conf_intervals[i] for i in range(len(means))]
        
        # Create slightly offset x positions
        offset = (idx - (len(label_stats)-1)/2) * offset_step
        x_positions = [layer + offset for layer in layers]
        #x_positions = layers
        # Plot the mean line with markers
        ax.plot(x_positions, means, '-o', linewidth=2, markersize=8, 
                 color=colors[idx], label=f'Label {map_lab[label]}')
        
        # Fill the confidence interval with distinct patterns
        ax.fill_between(x_positions, lower_bound, upper_bound, 
                         alpha=0.2, color=colors[idx], 
                         hatch=['////', '\\\\\\\\', '.', '*', 'x', '+'][idx % 6],
                         edgecolor='black', linewidth=0.5)
    
    # Improve the overall appearance
    ax.set_title(f'f1-scores for reg_lambda={reg_lambda}', fontsize=16)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Customize x-ticks to match the actual layer numbers
    ax.set_xticks(layers)
    
    # Add a legend with a semi-transparent background in a good position
    ax.legend(fontsize=12, framealpha=0.8, loc='best')
    

    
