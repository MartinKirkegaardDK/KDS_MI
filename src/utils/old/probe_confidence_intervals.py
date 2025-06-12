from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from copy import deepcopy
import random
from scipy.stats import sem, t
import numpy as np
from src.classes.datahandling import ActivationDataset
from src.classes.models import ClassificationProbe
from src.classes.hook_manager import HookManager
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



def plot_confidence_intervals(results,meta_data, map_lab):
    label_stats = defaultdict(lambda: defaultdict(list))
    
    # Collect all values per label per layer
    for layer, samples in results.items():
        for sample in samples:
            for label, value in sample.items():
                label_stats[label][layer].append(value)
    
    plt.figure(figsize=(12, 8))
    
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
        plt.plot(x_positions, means, '-o', linewidth=2, markersize=8, 
                 color=colors[idx], label=f'Label {map_lab[label]}')
        
        # Fill the confidence interval with distinct patterns
        plt.fill_between(x_positions, lower_bound, upper_bound, 
                         alpha=0.2, color=colors[idx], 
                         hatch=['////', '\\\\\\\\', '.', '*', 'x', '+'][idx % 6],
                         edgecolor='black', linewidth=0.5)
    
    # Improve the overall appearance
    plt.title('Confidence Intervals Across Layers', fontsize=16)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize x-ticks to match the actual layer numbers
    plt.xticks(layers)
    
    # Add a legend with a semi-transparent background in a good position
    legend = plt.legend(fontsize=12, framealpha=0.8, loc='best')
    
    # Add a border to the plot
    plt.box(True)
    plt.ylim(0,1)
    
    plt.tight_layout()
    plt.savefig(f'results/probe_confidence_intervals/{meta_data["model_name"]}_reg_lambda_{meta_data["reg_lambda"]}.png', bbox_inches='tight')

    plt.show()

def train_probe(meta_data,probe_by_layer,act_loader_by_layer, device):


    for layer, probe in probe_by_layer.items():
        act_loader = act_loader_by_layer[layer]
        optimizer = torch.optim.Adam(probe.parameters(), lr=meta_data["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(meta_data["amount_epochs"]):
            for act, label in act_loader:

                        
                label = label.to(device)
                batch_size = label.shape[0]
                #This just fixed the batch slicing
                if batch_size != 32:
                    break
                outputs = probe(act.to(device))
                preds = torch.argmax(outputs, dim=1)  # Get predicted class indices

                # Store labels and predictions (keep them on device)
                probe.all_preds.append(preds)
                probe.all_labels.append(label)

                loss = loss_fn(outputs, label.to(device))
                loss += meta_data["reg_lambda"] * sum(torch.norm(param, 2) for param in probe.parameters())

                accuracy = ((torch.argmax(outputs.detach(), dim=1) == label.to(device)).sum() / batch_size).item()
                #print('acc: ', accuracy, end='\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #print(loss)
        probe.compute_scores()


def model_setup(model_name:str) -> tuple[AutoModelForCausalLM,AutoTokenizer, str]:
    """loads a huggingface model

    Args:
        model_name (str): the huggingface name of a model. Example: AI-Sweden-Models/gpt-sw3-356m

    Returns:
        tuple[AutoModelForCausalLM,AutoTokenizer, str]: model, tokenizer, device
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    print("found device:",device)
    return model, tokenizer, device


def get_activations(meta_data: dict, 
    loader: DataLoader, 
    tokenizer: AutoTokenizer, 
    device: str,
    model:AutoModelForCausalLM,
    label_map=None
    ) -> dict: 

    if label_map == None:
        label_map = loader.dataset.label_map
    res_stream_act_by_layer = dict()
    activation_ds_by_layer = {
        layer: ActivationDataset(label_map=label_map)
        for layer in range(meta_data["hidden_layers"])
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    for ind, (text, label) in enumerate(tqdm(loader)):

        if ind > 5:
            break

        
        tokenized = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with HookManager(model) as hook_manager:
            for layer in range(meta_data["hidden_layers"]):
                res_stream_act_by_layer[layer] = hook_manager.attach_residstream_hook(
                    layer=layer,
                    pre_mlp=False,
                    pythia=True if isinstance(model, GPTNeoXForCausalLM) else False
                )

            model(**tokenized)

        # flattening [batch, pad_size, ...] to [tokens, ...]
        attn_mask = tokenized.attention_mask.flatten() # [tokens]
        label = label.unsqueeze(-1).expand(-1, tokenized.attention_mask.shape[1]).flatten() # [tokens]

        for layer in range(meta_data["hidden_layers"]):
            res_stream_act_by_layer[layer] = res_stream_act_by_layer[layer][0].view(-1, meta_data["hidden_size"]) # [tokens, hidden_size]
            activation_ds_by_layer[layer].add_with_mask(res_stream_act_by_layer[layer], label, attn_mask)
    return activation_ds_by_layer


def create_classes_by_layer(meta_data: dict, activation_ds_by_layer: dict, device: str):
    probe_by_layer = {
        layer: ClassificationProbe(in_dim=meta_data["hidden_size"], num_labs=meta_data["number_labels"], device=device)
        for layer in range(meta_data["hidden_layers"])
    }

    act_loader_by_layer = {
        layer: DataLoader(activation_ds_by_layer[layer], batch_size=32, shuffle=True)
        for layer in range(meta_data["hidden_layers"])
    }
    return probe_by_layer, act_loader_by_layer

def create_bootstrap_dataset(activation_ds_by_layer):

    copy_dataset = deepcopy(activation_ds_by_layer)

    for layer in copy_dataset.keys():
        activations = copy_dataset[layer].predictors
        labels = copy_dataset[layer].labels
        
        len_dataset = len(activations)
        indicies = random.choices(range(len_dataset),k=len_dataset)

        new_acts = [activations[index] for index in indicies]
        new_labels = [labels[index] for index in indicies]

        copy_dataset[layer].predictors = new_acts
        copy_dataset[layer].labels = new_labels
    return copy_dataset

def bootstrap(n, meta_data,activation_ds_by_layer, device):
    li = []
    for i in range(n):
       #print(activation_ds_by_layer,"grr")
        new_activation_ds_by_layer = create_bootstrap_dataset(activation_ds_by_layer)
        probe_by_layer, act_loader_by_layer = create_classes_by_layer(meta_data, new_activation_ds_by_layer, device)
        train_probe(meta_data, probe_by_layer,act_loader_by_layer, device)
        li.append(probe_by_layer)
    return li
    