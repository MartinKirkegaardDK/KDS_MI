from matplotlib import pyplot as plt
import pickle
import numpy as np

def main(num_layers:int, hook_points:list, model_name_temp:str):
    
    
    with open(f'results/data/neuron_contributions/{model_name_temp}.pkl', 'rb') as f:
        neuron_contributions = pickle.load(f)
 

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
            # Highlight neurons with c*ontribution > 4 × average
            avg = contributions.mean()
            for neuron_idx, value in enumerate(contributions):
                if value > 2.5 * avg:
                    axs[i].text(
                        neuron_idx, value, f'N{neuron_idx}',
                        ha='center', va='bottom', fontsize=8, rotation=90
                    )

        plt.tight_layout()
        plt.savefig(f'results/neuron_contributions/{model_name_temp}_layer_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Free memory
