import json
from collections import defaultdict

from statistics import mean
from matplotlib import pyplot as plt
from src.utils.new.compatibility import *
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import sem, t

def load_probe_results():

    lambdas = [0.1, 0.5, 1, 2, 5, 10]

    layers = list(range(24))

    path = lambda ha, rl : f'results/data/probe_confidence_intervals/gpt_sw3_356m_{ha}_reg_lambda_{rl}.json'

    data = defaultdict(dict)
    for hook_address in HookAddress:
        for r_lambda in lambdas:
            with open(path(hook_address.address, r_lambda)) as file:
                data_ = json.load(file)

            map_label = data_['map_label']

            for layer in layers:
                for bootstrap in data_[str(layer)]:
                    for lab in map_label.keys():
                        bootstrap[map_label[lab]] = bootstrap[lab]
                        del bootstrap[lab]

                data_[layer] = data_[str(layer)]
                del data_[str(layer)]
                    
            del data_['map_label']

            data[hook_address.address][r_lambda] = data_

    return data




def make_big_plot():

    data = load_probe_results()

    r_lambdas = [0.1, 0.5, 1, 2, 5, 10]
    hook_addresses = list(HookAddress)
    layers = list(range(24))

    def color_spec(color):
        rgb = mcolors.hex2color(color)

        # make spectrum from white to your color
        n_steps = len(r_lambdas)
        whites = np.ones((n_steps, 3))
        colors = np.array([rgb] * n_steps)
        alphas = np.linspace(0.25, 1, n_steps).reshape(-1, 1)

        spectrum = whites * (1 - alphas) + colors * alphas

        return spectrum

    fig, axs = plt.subplots(len(hook_addresses),2, figsize=(10, 2 * len(hook_addresses)), sharex=True)
    axs = axs.T.flatten()

    for idx_ax, hook_address in enumerate(hook_addresses):

        ax = axs[idx_ax]

        face_spectrum = color_spec(Colors.face(hook_address))
        outline_spectrum = color_spec(Colors.outline(hook_address))

        for idx, r_lambda in enumerate(r_lambdas):
            current = data[hook_address.address][r_lambda]

            y = []
            for layer in layers:
                bootstraps = []
                for bootstrap in current[layer]:
                    bootstraps.append(mean(bootstrap.values()))
                y.append(mean(bootstraps))


            ax.fill_between(layers, y,color=face_spectrum[idx])
            ax.plot(layers, y, color=outline_spectrum[idx], linewidth=1)

        sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(outline_spectrum), norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='regularization')

        tick_positions = np.linspace(0, 1, len(r_lambdas))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(r_lambdas)
        cbar.ax.invert_yaxis()

        ax.set_ylim((0, 0.9))
        ax.set_title(hook_address.address)
        ax.set_ylabel('f1 score')
        if idx_ax + 1 == len(hook_addresses):
            ax.set_xlabel('layer')


    for idx_ax, hook_address in enumerate(hook_addresses, start=len(hook_addresses)):

        ax = axs[idx_ax]

        accuracies_by_layer = data[hook_address.address][2]

        label_stats = defaultdict(lambda: defaultdict(list))
        
        # Collect all values per label per layer
        for layer, samples in accuracies_by_layer.items():
            for sample in samples:
                for label, value in sample.items():
                    label_stats[label][int(layer)].append(value)
        
        
        # Use a distinct color palette
        colors = plt.cm.Set1(range(5))
        colors_by_label = {
            'da': colors[0],
            'en': colors[1],
            'is': colors[2],
            'sv': colors[3],
            'nb': colors[4]
        }
        
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
            if idx_ax - len(hook_addresses) == 0:
                ax.plot(x_positions, means, linewidth=1, 
                        color=colors_by_label[label], label=f'{label}')
            else:
                ax.plot(x_positions, means, linewidth=1, 
                        color=colors_by_label[label])
            
            # Fill the confidence interval with distinct patterns
            ax.fill_between(x_positions, lower_bound, upper_bound, 
                            alpha=0.2, color=colors_by_label[label], 
                            #hatch=['////', '\\\\\\\\', '.', '*', 'x', '+'][idx % 6],
                            #edgecolor='black', 
                            linewidth=0.5)
            
            ax.set_ylim((0, 0.9))
            ax.set_title(f'{hook_address.address}, regularization={2}')
            ax.set_ylabel('f1 score')
            if idx_ax + 1 == len(hook_addresses) * 2:
                ax.set_xlabel('layer')


    fig.legend(bbox_to_anchor=(0.98, 0.011), loc='upper right', ncol=5, title='language')
    fig.tight_layout()

    fig.savefig('results/figures/probes/big_fig.png', bbox_inches='tight', dpi=400)