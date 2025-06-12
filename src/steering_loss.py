import torch
from transformers import PreTrainedTokenizerBase, GPTNeoXForCausalLM, GPT2Model
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils.new.data import AntibioticDataset, ParallelNSPDataset, load_bible_data, load_steering_vector
from src.utils.new.hooking import HookManager
from src.utils.new.compatibility import Device, HookAddress, ModelConfig, Colors


from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from collections import defaultdict
from transformers import PreTrainedTokenizerBase, GPTNeoXForCausalLM, GPT2Model, AutoModelForCausalLM, AutoTokenizer
from statistics import mean


def generate_with_steering(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        hook_address,
        text_prompts: AntibioticDataset | list[str] | str,
        steering_vector: torch.Tensor,
        steering_lambda: int = 1,
        amount_samples: int = 10,
        cut_off: int = 10
    ) -> list[str]:
    '''
    Generates text from a set of prompts. The prompts will be cut up after a certain amount of tokens, and continued by the model under steering. If prompts are provided as TextClassificationDataset, they are automatically cut up randomly.

    Args:
        model: the torch model
        tokenizer: the model's tokenizer
        text_prompts: a TextClassificationDataset containing the prompts OR a list of prompts OR one prompt as str
        steering_vector: the torch.Tensor steering vector
        steering_lambda: the scalar which will scale the steering vector before it being applied
        amount_sample: the amount of sentence generations to perform
        cut_off: at what token the prompts will be cut off

    Returns:
        a list of the output strings
    '''

    device = Device.device(model)

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    is_textclassdataset = isinstance(text_prompts, Dataset)
    if type(text_prompts) == str:
        text_prompts = [text_prompts]
    outputs = []

    with HookManager(model) as hook_manager:

        hook_manager.steer(
            hook_address=hook_address,
            steering_vector=steering_vector.to(device),
            scalar=steering_lambda,
        )
        
        for i in range(min(amount_samples, len(text_prompts))):
            if is_textclassdataset:
                text, _ = text_prompts[i]
            else:
                text = text_prompts[i]
    
            tokenized = tokenizer(text, return_tensors='pt').to(device)

            undecoded_output = model.generate(
                inputs=tokenized.input_ids[:, :cut_off] if is_textclassdataset else tokenized.input_ids, 
                max_length=100, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True,
                #attention_mask=tokenized.attention_mask,
                #pad_token_id=tokenizer.pad_token_id

            )

            outputs.append(tokenizer.decode(undecoded_output[0]).replace('\n', '  '))

    return outputs


def loss_with_steering(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        hook_address,
        prompt: str,
        continuation: str,
        steering_vector: torch.Tensor,
        steering_lambda: int = 1
    ) -> float:
    '''
    Computes the loss (next-token prediction) for a steered model, given a prompt and a true continuation. Returns only the loss on the continuation.

    Args:
        model: the torch model
        tokenizer: the model's tokenizer        prompt: the text prompt, given as context for prediction
        continuation: the ground truth continuation, used for computing loss
        steering_vector: the torch.Tensor steering vector
        steering_lambda: the scalar which will scale the steering vector before it being applied

    Returns:
        Average loss on continuation
    '''
    
    device = Device.device(model)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token


    with HookManager(model) as hook_manager:
        hook_manager.steer(
            hook_address=hook_address,
            steering_vector=steering_vector.to(device),
            scalar=steering_lambda,
        )

        # tokenizes the prompt and the continuation
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
        tokenized_continuation = tokenizer(continuation, return_tensors='pt').to(device)

        # combines them into one input
        tokenized_combined = torch.concat(
            (tokenized_prompt.input_ids, tokenized_continuation.input_ids), 
            dim=1
        )

        # computes output from model 
        output = model(tokenized_combined)

        # gets predictions and labels (need to be shifted by one)
        shift_logits = output.logits[:, :-1, :]
        shift_labels = tokenized_combined[:, 1:]

        # computes cross-entropy loss for all tokens individualle
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_by_token = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # isolates tokens from continuation and averages their loss
        loss_continuation = loss_by_token[-tokenized_continuation.input_ids.numel():]
        avg_loss_continuation = loss_continuation.mean().item()

        return avg_loss_continuation


    
    

def plot_steering_vector_scores(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        ds: ParallelNSPDataset,
        steering_lambdas: int | list,
        lan1: str,
        lan2: str,
        amount_datapoints: int,
        layers=None,
        hook_addresses=None,
        out_file=None
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

    losses_with_steering = defaultdict(lambda: defaultdict(list))
    losses_without_steering = defaultdict(lambda: defaultdict(list))
    losses_with_correct_context = defaultdict(lambda: defaultdict(list))

    if not hook_addresses:
        hook_addresses = list(HookAddress)
    if not layers:
        layers = list(range(ModelConfig.hidden_layers(model)))
    if type(steering_lambdas) != list:
        steering_lambdas = [steering_lambdas]


    for idx, x in tqdm(enumerate(ds)):
        if idx > amount_datapoints:
            break
        
        for layer in layers:
            for hook_address in hook_addresses:
                for steering_lambda in steering_lambdas:

                    steering_vector = load_steering_vector(lan2, hook_address.layer(layer), model)
                
                    # computes the loss, when the model is steered towards the continuation language 
                    loss_with_steering_ = loss_with_steering(
                        model=model,
                        tokenizer=tokenizer,
                        hook_address=hook_address.layer(layer),
                        prompt=x[lan1][0],
                        continuation=x[lan2][1],
                        steering_vector=steering_vector,
                        steering_lambda=steering_lambda
                    )
                    losses_with_steering[hook_address.layer(layer)][steering_lambda].append(loss_with_steering_)

                    # computes the loss, when the model is not steering
                    loss_without_steering = loss_with_steering(
                        model=model,
                        tokenizer=tokenizer,
                        hook_address=hook_address.layer(layer),
                        prompt=x[lan1][0],
                        continuation=x[lan2][1],
                        steering_vector=steering_vector,
                        steering_lambda=0
                    )
                    losses_without_steering[hook_address.layer(layer)][steering_lambda].append(loss_without_steering)

                    # computes the loss without steering, but where the prompt matches the continuation language
                    loss_with_correct_context = loss_with_steering(
                        model=model,
                        tokenizer=tokenizer,
                        hook_address=hook_address.layer(layer),
                        prompt=x[lan2][0],
                        continuation=x[lan2][1],
                        steering_vector=steering_vector,
                        steering_lambda=0
                    )
                    losses_with_correct_context[hook_address.layer(layer)][steering_lambda].append(loss_with_correct_context)


    avg_normalized_improvement = defaultdict(dict)

    for layer in layers:
        for hook_address in hook_addresses:
            for steering_lambda in steering_lambdas:
                avg_steered = mean(losses_with_steering[hook_address.layer(layer)][steering_lambda])
                avg_non_steered = mean(losses_without_steering[hook_address.layer(layer)][steering_lambda])
                avg_with_context = mean(losses_with_correct_context[hook_address.layer(layer)][steering_lambda])

                avg_steering_improvement = avg_non_steered - avg_steered
                avg_context_improvement = avg_non_steered - avg_with_context

                avg_normalized_improvement[hook_address.layer(layer)][steering_lambda] = avg_steering_improvement / avg_context_improvement


    fig, axs = plt.subplots(
        len(steering_lambdas), 
        len(hook_addresses), 
        figsize=(3.5 * len(hook_addresses), 2.3 * len(steering_lambdas)),
        sharex=True,
        sharey=True)
                            
    if type(axs) == Axes:
        axs = np.array([[axs]])
    elif len(steering_lambdas) == 1:
        axs = np.array([axs])
    elif len(hook_addresses) == 1:
        axs = np.array([[ax] for ax in axs])

    for idx, steering_lambda in enumerate(steering_lambdas):
        for idy, hook_address in enumerate(hook_addresses):
            ax = axs[idx][idy]
            improvement_scores = [avg_normalized_improvement[hook_address.layer(layer)][steering_lambda] for layer in layers]

            colors = [Colors.face(hook_address) if val > 0 else '#eaeaea' for val in improvement_scores]
            edgecolors = [Colors.outline(hook_address) if val > 0 else '#d0d0d0' for val in improvement_scores]

            ax.bar(layers, improvement_scores, color=colors, edgecolor=edgecolors)
            ax.set_title(f"{hook_address.address}, lambda={steering_lambda}")
            ax.set_ylim(-0.3, 0.5)
            ax.axhline(y=0, color='k', linestyle='dotted', linewidth=1)

    fig.supxlabel("layer")
    fig.supylabel("score")
    fig.tight_layout()

    if out_file:
        fig.savefig(f'{out_file}', transparent=False, dpi=300)

    return avg_normalized_improvement



def first_fig(model, tokenizer, shuffled_dataset, out_file=None):

    scores = plot_steering_vector_scores(
        model=model,
        tokenizer=tokenizer,
        ds=shuffled_dataset,
        steering_lambdas=[5],
        lan1='en',
        lan2='da',
        amount_datapoints=50
    )


    fig, axs = plt.subplots(
    2, 
    3, 
    figsize=(3.5 * 3, 2.3 * 2),
    sharex=True,
    sharey=True)
                        
    axs = axs.flatten()

    layers = list(range(ModelConfig.hidden_layers(model)))
    hook_addresses = list(HookAddress)

    for idx, hook_address in enumerate(hook_addresses):
        ax = axs[idx]
        improvement_scores = [scores[hook_address.layer(layer)][5] for layer in layers]

        colors = [Colors.face(hook_address) if val > 0 else '#eaeaea' for val in improvement_scores]
        edgecolors = [Colors.outline(hook_address) if val > 0 else '#d0d0d0' for val in improvement_scores]

        ax.bar(layers, improvement_scores, color=colors, edgecolor=edgecolors)
        ax.set_title(f"{hook_address.address}, lambda={5}")
        ax.set_ylim(-0.3, 0.5)
        ax.axhline(y=0, color='k', linestyle='dotted', linewidth=1)

    fig.supxlabel("layer")
    fig.supylabel("score")
    fig.tight_layout()

    if out_file:
        fig.savefig(f'{out_file}', transparent=False, dpi=300)



def main(
        model_name,
        model_name_temp,
        lan1,
        lan2,
        device
    ):
    """
    Plots the steering loss and saves it at: results/steering_loss/*model_name*
    """
    saved_path = "results/steering_loss"

    # loads model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # loads data
    ds = load_bible_data(lan1, lan2)


    steering_vector_folder = Path(steering_vector_folder)
    num_layers = model.config.num_hidden_layers
    steering_vectors_by_layer = {
        layer: load_steering_vector
        for layer in range(num_layers)
    }

    steering_lambdas = [1, 2, 5, 10, 15]
    Path(saved_path).mkdir(parents=True, exist_ok=True)

    if "download" in model_name:
        model_name = model_name.split("/")[-1]

    for steering_lambda in steering_lambdas:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))  # One figure per lambda
        plot_loss_for_steering_vectors(
            model,
            tokenizer,
            ds,
            steering_vectors_by_layer,
            steering_lambda,
            lan1=lan1,
            lan2=lan2,
            amount_datapoints=100,
            ax=ax
        )
        fig.tight_layout()
        fig.savefig(f"{saved_path}/{model_name_temp}_lambda_{steering_lambda}.png")
        plt.close(fig)  # Important to prevent memory issues in long runs