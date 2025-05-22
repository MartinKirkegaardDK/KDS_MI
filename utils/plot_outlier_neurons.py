from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib.ticker as ticker

def plot_neuron_contributions(num_layers:int, hook_points:list, model_name_temp:str, neuron_contributions:dict):
    d = defaultdict(int)


    for layer in range(num_layers):
        fig, axs = plt.subplots(1, len(hook_points), figsize=(20, 4))
        axs = axs.flatten()

        for i, hook in enumerate(hook_points):
            contributions = neuron_contributions[(layer, hook)]
            x = np.arange(len(contributions))
            axs[i].bar(x, contributions)
            axs[i].set_title(f'{hook}')
            axs[i].set_xlabel('Neuron Index')
            axs[i].set_ylabel('Sum of Activations')
            axs[i].set_ylim(0, max(contributions) * 1.3)
            bars = axs[i].bar(x, contributions, color='gray')  # base color

            avg = contributions.mean()
            for neuron_idx, value in enumerate(contributions):
                if value > 3 * avg:
                    bars[neuron_idx].set_edgecolor('red')
                    bars[neuron_idx].set_linewidth(1.1)  # Thicker edge
                    bars[neuron_idx].set_zorder(10)  # Make sure it’s not hidden
                    axs[i].text(
                        neuron_idx, value, f'N{neuron_idx}',
                        ha='center', va='bottom', fontsize=8, rotation=90,
                        zorder=20  # ensure it's in front of the bar edge
                    )
                    d[neuron_idx] += 1
        plt.tight_layout()
        plt.savefig(f'results/neuron_contributions/{model_name_temp}_layer_{layer}.png', dpi=550, bbox_inches='tight')
        plt.close(fig)  # Free memory


    plt.figure(figsize=(10, 5))

    # Convert the dict to sorted lists for plotting
    neuron_indices = list(d.keys())
    occurrences = [d[k] for k in neuron_indices]

    bars = plt.bar(neuron_indices, occurrences, color="gray", width=1.2)

    # Extend y-axis limit for text space
    max_occ = max(occurrences)
    plt.ylim(0, max_occ * 1.3)

    # Add integer-only y-axis ticks
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Add horizontal gridlines
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Annotate each bar with its neuron index
    for idx, bar in zip(neuron_indices, bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'N{idx}',
                ha='center', va='bottom', fontsize=8, rotation=90, zorder=10)

    plt.title("Outlier Neurons Distributions")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron occurrence")
    plt.tight_layout()
    plt.savefig(fr"results\outlier_neuron_distributions\{model_name_temp}.png", bbox_inches="tight")

def common_outliers(model_name_temp,obj):



    common_indices = obj["common_indices"]
    top_idx_counter = obj["top_idx_counter"]
    bot_idx_counter = obj["bot_idx_counter"]



    data = []
    for idx in common_indices:
        data.append({
            "index": idx,
            "top_count": top_idx_counter[idx],
            "bottom_count": bot_idx_counter[idx]
        })

    df = pd.DataFrame(data).sort_values("top_count", ascending=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(df["index"].astype(str), df["top_count"], label="Top Activations")
    plt.bar(df["index"].astype(str), df["bottom_count"], label="Bottom Activations", alpha=0.7)
    plt.xlabel("Neuron Index")
    plt.ylabel("Frequency")
    plt.title("Overlap Between Top and Bottom Activating Neurons")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"results/outlier_neuron_distributions/{model_name_temp}.png")
