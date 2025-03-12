import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from collections import defaultdict
from transformers import PreTrainedTokenizerBase, GPTNeoXForCausalLM, GPT2Model
from statistics import mean

from classes.datahandling import ParallelNSPDataset
from utils.steering import loss_with_steering

def plot_PCA(steering_vectors: dict[str, torch.Tensor]) -> None:
    '''
    plots steering vector on a 2d PCA plot

    Args:
        steering_vectors: a dictionary with keys as languages, and values as torch Tensor steering vectors
    '''

    languages, vectors = zip(*steering_vectors.items())
    vectors = torch.stack(vectors).cpu().numpy()

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(vectors)

    x, y = transformed.T 
    scatter = plt.scatter(x, y, c=range(len(x)), cmap='viridis')
    plt.legend(handles=scatter.legend_elements()[0], labels=languages)


def plot_activations_PCA(activations_by_language: dict[str, list[torch.Tensor]]) -> None:
    '''
    plots activations on a 2d PCA plot

    Args:
        activations_by_language: dictionary with keys as languages and values as lists of activation vectors 
    '''

    languages = []
    for language, activations in activations_by_language.items():
        languages.extend([language] * len(activations))

    mapping = {language: idx for idx, language in enumerate(activations_by_language.keys())}
    reverse_mapping = {idx: language for language, idx in mapping.items()}

    vectors = torch.concat(
        tensors=[
            torch.stack(activations, dim=0)
            for activations in activations_by_language.values()
        ],
        dim=0
    )

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(vectors)


    x, y = transformed.T 
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=list(map(lambda x: mapping[x], languages)))


    plt.legend(handles=scatter.legend_elements()[0], labels=[reverse_mapping[i] for i in range(len(reverse_mapping))])



def plot_loss_for_steering_vectors(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        ds: ParallelNSPDataset,
        steering_vectors_by_layer: dict[int, torch.Tensor],
        steering_lambda: int,
        lan1: str,
        lan2: str,
        amount_datapoints: int,
    ):
    '''
    plots loss for steering each layer

    Args:
        model: model to steer,
        tokenizer: the model's tokenizer
        ds: dataset of sentence, continuation for two parallel languages
        steering_vectors_by_layer: dictionary with keys as layer and values as corresponding steering vector for the given language
        lan1: language to steer *away from*
        lan2: language to steer *toward*
        amount_datapoints: only compute on first k datapoints in dataset

    Returns:
        a pyplot figure
    '''

    losses_with_steering = defaultdict(list)
    losses_without_steering = defaultdict(list)
    losses_with_correct_context = defaultdict(list)

    for idx, x in enumerate(ds):
        if idx > amount_datapoints:
            break
        
        for layer, steering_vector in steering_vectors_by_layer.items():
            
            # computes the loss, when the model is steered towards the continuation language 
            loss_with_steering_ = loss_with_steering(
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                prompt=x[lan1][0],
                continuation=x[lan2][1],
                steering_vector=steering_vector,
                steering_lambda=steering_lambda
            )
            losses_with_steering[layer].append(loss_with_steering_)

            # computes the loss, when the model is not steering
            loss_without_steering = loss_with_steering(
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                prompt=x[lan1][0],
                continuation=x[lan2][1],
                steering_vector=steering_vector,
                steering_lambda=0
            )
            losses_without_steering[layer].append(loss_without_steering)

            # computes the loss without steering, but where the prompt matches the continuation language
            loss_with_correct_context = loss_with_steering(
                model=model,
                tokenizer=tokenizer,
                layer=layer,
                prompt=x[lan2][0],
                continuation=x[lan2][1],
                steering_vector=steering_vector,
                steering_lambda=0
            )
            losses_with_correct_context[layer].append(loss_with_correct_context)

    fig, ax = plt.subplots(1, 1, figsize=(10,5))

    layers = list(losses_with_steering.keys())
    avg_normalized_improvement = []


    for layer in layers:
        avg_steered = mean(losses_with_steering[layer])
        avg_non_steered = mean(losses_without_steering[layer])
        avg_with_context = mean(losses_with_correct_context[layer])

        avg_steering_improvement = avg_non_steered - avg_steered
        avg_context_improvement = avg_non_steered - avg_with_context

        avg_normalized_improvement.append(avg_steering_improvement / avg_context_improvement)

    ax.bar(layers, avg_normalized_improvement)

    plt.show()

    