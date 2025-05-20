import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from matplotlib import pyplot as plt
from copy import deepcopy
import random
from scipy.stats import sem, t
from tqdm import tqdm

from collections import defaultdict
import sklearn.metrics as metrics

from refactor.utils.data import FilePaths, load_antibiotic_data
from refactor.utils.hooking import get_activations
from refactor.utils.compatibility import ModelConfig

class ClassificationProbe(nn.Module):

    def __init__(self, in_dim, num_labs, device):
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_dim,
            out_features=num_labs,
            device=device
        )
        self.device = device
        self.accuracy = None
        self.all_preds = []
        self.all_labels = []
    
    def compute_scores(self): 
        all_preds = torch.cat(self.all_preds).cpu()
        all_labels = torch.cat(self.all_labels).cpu()
    

        self.find_used_labels()
        self.accuracy = metrics.accuracy_score(all_labels, all_preds)
        self.classification_report = metrics.classification_report(all_labels, all_preds, output_dict = True, zero_division = 1)
        
        self.class_accuracies = self.class_accuracies(self.classification_report)
    
    def find_used_labels(self):
        
        self.used_labels = set()
        for labels in self.all_labels:
      
            unique_labels = set(np.array(labels.cpu()))
            [self.used_labels.add(x) for x in unique_labels]
        

    def class_accuracies(self, classification_report):
        d = dict()
        for label in self.used_labels:
            
            d[int(label)] = self.classification_report[str(label)]["f1-score"]
        return d 


    def forward(self, x):

        return torch.softmax(self.linear(x), dim=1)



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
                outputs = probe(act)
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
        new_activation_ds_by_layer = create_bootstrap_dataset(activation_ds_by_layer)
        probe_by_layer, act_loader_by_layer = create_classes_by_layer(meta_data, new_activation_ds_by_layer, device)
        train_probe(meta_data, probe_by_layer,act_loader_by_layer, device)
        li.append(probe_by_layer)
    return li



def main(model_name, reg_lambdas):
    """This function runs an entire pipeline that bootstraps, trains and creates confidence intervals showing
       The probes f1 score on different labels and across layers
       
       We bootstrap 10 times
       Results are saved in this folder: results/data/probe_confidence_intervals/*model_name*_reg_lambda_*reg_lambda*

    Args:
        model_name (_type_): _description_
        reg_lambdas (_type_): _description_
    """


    # loads model
    print("Load model")
    model, tokenizer, device = model_setup(model_name)


    # loads data
    print("Load data")
    ds = load_antibiotic_data(
        file_paths=FilePaths.antibiotic,
        file_extension='txt'
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    
    
    # sets training parameters
    meta_data = {}
    meta_data["hidden_size"] = ModelConfig.hidden_size
    meta_data["hidden_layers"] = ModelConfig.hidden_layers
    meta_data["model_name"] = model_name.split("/")[0]
    meta_data["learning_rate"] = 0.001
    meta_data["reg_lambda"] = 10
    meta_data["amount_epochs"] = 1


    # extracts activation from forward passes on data
    # We use hooks to extract the different layer activations that will be used to train our probes
    
    print("Extract activations")
    activations = get_activations(
        loader=loader, 
        model=model,
        tokenizer=tokenizer,
        hook_addresses=None,
        layers=None,
        max_batches=20,
        sampling_prob=0.1
    )