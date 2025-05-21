import torch
from torch.utils.data import DataLoader
import pickle
from refactor.utils.data import FilePaths, load_antibiotic_data
from refactor.utils.hooking import get_activations as get_activations_new
from refactor.probes import model_setup
import matplotlib.pyplot as plt
import numpy as np

def main(model_name,model_name_temp):


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


    print("Extract activations")
    activations = get_activations_new(
        loader=loader, 
        model=model,
        tokenizer=tokenizer,
        hook_addresses=None,
        layers=None,
        max_batches=20,
        sampling_prob=0.1
    )
    
    hook_points = ["attention:post","attention:pre","mlp:post","mlp:pre","layernorm_1:pre","layernorm_2:pre"]
    
    create_neuron_contributions(activations, hook_points,model_name_temp)
    


def create_neuron_contributions(activations:dict,hook_points:list, model_name_temp: str):
    
    num_layers = 5
    neuron_contributions = {}

    for layer in range(num_layers):
        for hook in hook_points:
            # Get activations for this specific layer and hook point
            activ_list = activations[f"layer.{layer}.{hook}"].predictors
            
            # Stack all tensors in activ_list
            stacked_activations = torch.stack(activ_list)
            
            # Take absolute values and sum across all dimensions except the neuron dimension
            # Assuming the last dimension is the neuron dimension
            neuron_sums = torch.abs(stacked_activations).sum(dim=0)
            
            # Store in our dictionary with a key that identifies both layer and hook
            key = (layer, hook)
            neuron_contributions[key] = neuron_sums.cpu().numpy()
            
    with open(f'results/data/neuron_contributions/{model_name_temp}.pkl', 'wb') as f:
        pickle.dump(neuron_contributions, f)


