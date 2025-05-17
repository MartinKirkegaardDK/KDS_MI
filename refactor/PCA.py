import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from utils.data import load_antibiotic_data, FilePaths
from utils.hooking import get_activations

from utils.compatibility import Hookpoints, ModelConfig, HookAddress

def compute_PCA(activations):

    languages = activations.label_map.keys()

    mean_vectors = [
        torch.mean(activations.filter_by_language(language, return_tensors=True), dim=0)
        for language in languages
    ]

    pca = PCA(n_components=2).fit(torch.stack(mean_vectors).cpu())

    transformed = {
        language: pca.transform(activations.filter_by_language(language, return_tensors=True).cpu())
        for language in languages    
    }

    return transformed


def compute_PCA_2(activations):

    languages = activations.label_map.keys()

    vectors = [
        activations.filter_by_language(language, return_tensors=True)
        for language in languages
    ]

    pca = PCA(n_components=2).fit(torch.cat(vectors, dim=0).cpu())

    transformed = {
        language: pca.transform(activations.filter_by_language(language, return_tensors=True).cpu())
        for language in languages    
    }

    return transformed


def compute_PCA_2(activations):

    languages = activations.label_map.keys()


    vectors = [
        activations.filter_by_language(language, return_tensors=True)
        for language in languages
    ]

    pca = PCA(n_components=2).fit(torch.cat(vectors, dim=0).cpu())

    transformed = {
        language: pca.transform(activations.filter_by_language(language, return_tensors=True).cpu())
        for language in languages    
    }

    return transformed





def plot_PCA(transformed, ax, colors=None):

    languages = transformed.keys()

    if not colors:
        cmap = plt.cm.Set1(range(len(languages)))
        colors = {
            language: cmap[idx]
            for idx, language in enumerate(languages)
        }

    
    # plots the transformed activations
    for language in languages:
        x, y = transformed[language].T 
        ax.scatter(
            x, 
            y, 
            color=colors[language],
            alpha=0.4, 
            s=8,
            label=language
        )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend()


def main(model_url, device):
    model = AutoModelForCausalLM.from_pretrained(model_url).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    loader = DataLoader(load_antibiotic_data(file_paths=FilePaths.antibiotic), shuffle=True, batch_size=32)

    layers = list(range(ModelConfig.hidden_layers(model)))
    hook_addresses = list(HookAddress)

    fig, axs = plt.subplots(len(layers), len(hook_addresses), figsize=(20, 80))

    activations = get_activations(
        loader,
        model,
        tokenizer,
        layers=layers,
        hook_addresses=hook_addresses,
        max_batches=10
    )

    for idx, layer in enumerate(layers):
        for idy, hook_address in enumerate(hook_addresses):
            transformed = compute_PCA_2(
                activations=activations[hook_address.layer(layer)]
            )

            plot_PCA(transformed, axs[idx][idy])

    






    






if __name__ == '__main__':
    main(model_url='EleutherAI/pythia-14m', device='cpu')