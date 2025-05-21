import torch
from transformers import PreTrainedTokenizerBase, GPTNeoXForCausalLM, GPT2Model
from torch.utils.data import Dataset
from pathlib import Path

from utils.data import TextClassificationDataset, ParallelNSPDataset, load_bible_data, load_steering_vector
from utils.hooking import HookManager
from utils.compatibility import Device, HookAddress


import torch
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
        text_prompts: TextClassificationDataset | list[str] | str,
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
    

def plot_loss_for_steering_vectors(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        ds: ParallelNSPDataset,
        layers,
        hook_address,
        steering_lambda: int,
        lan1: str,
        lan2: str,
        amount_datapoints: int,
        ax: Axes
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


    steering_vector = load_steering_vector(lan2, hook_address, model)

    for idx, x in enumerate(ds):
        if idx > amount_datapoints:
            break
        
        for layer in layers:
            
            # computes the loss, when the model is steered towards the continuation language 
            loss_with_steering_ = loss_with_steering(
                model=model,
                tokenizer=tokenizer,
                hook_address=hook_address,
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
                hook_address=hook_address,
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
                hook_address=hook_address,
                prompt=x[lan2][0],
                continuation=x[lan2][1],
                steering_vector=steering_vector,
                steering_lambda=0
            )
            losses_with_correct_context[layer].append(loss_with_correct_context)


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
    ax.set_ylabel("relative improvement in loss")
    ax.set_xlabel("layer")
    ax.set_title(f"steering lambda={steering_lambda}")
    ax.set_ylim(-2.5, 1)



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