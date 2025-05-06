
#import seaborn as sns
from torch.utils.data import DataLoader
from pathlib import Path
import json

from utils.probe_confidence_intervals import model_setup, bootstrap, plot_confidence_intervals,get_activations
from utils.probe_confidence_intervals import create_classes_by_layer
from utils.preprocessing import load_txt_data

import numpy as np
from collections import defaultdict


def run(model_name, reg_lambdas):
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
    data_folder = Path('data/preprocessed/train')
    print("Load data")
    ds = load_txt_data(
        file_paths={
            'da': data_folder / 'da.txt',
            'en': data_folder / 'en.txt',
            'sv': data_folder / 'sv.txt',
            'nb': data_folder / 'nb.txt',
            'is': data_folder / 'is.txt'
        },
        file_extension='txt'
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)



    # sets training parameters
    meta_data = {}

    try:
        meta_data["hidden_size"] = model.config.n_embd
    except AttributeError:
        meta_data["hidden_size"] = model.config.hidden_size

    meta_data["hidden_layers"] = model.config.num_hidden_layers
    meta_data["model_name"] = model_name.split("/")[0]
    meta_data["learning_rate"] = 0.001
    meta_data["reg_lambda"] = 10
    meta_data["amount_epochs"] = 1


    # extracts activation from forward passes on data
    # We use hooks to extract the different layer activations that will be used to train our probes
    
    print("Extract activations")
    activation_ds_by_layer = get_activations(meta_data,loader, tokenizer, device, model)

    s = set()
    for i in range(meta_data["hidden_layers"]):
        unique_labels = set(np.array(activation_ds_by_layer[i].labels))
        [s.add(x) for x in unique_labels]
    number_labels = len(s)
    meta_data["number_labels"] = number_labels




    
    data_output_folder = Path('results/data/probe_confidence_intervals')

    for reg_lambda in reg_lambdas:
        meta_data['reg_lambda'] = reg_lambda

        # initiates all the probes and corresponding data loaders
        #print("Initiate classes")
        #probe_by_layer, act_loader_by_layer = create_classes_by_layer(meta_data, activation_ds_by_layer, device)


        # We bootstrap the data and train 10 probes for each layer in order to get "confidence intervals"
        print("Bootstrap")
        map_lab = ds.map_label
        boot = bootstrap(10, meta_data, activation_ds_by_layer, device)
        d = defaultdict(list)
        for run in boot:
            for layer in run.keys():
                class_accuracies = run[layer].class_accuracies
                d[layer].append(class_accuracies)



        # saves data used in plots
        
        d['map_label'] = map_lab
        if "downloaded" in model_name:
            model_name = model_name.split("/")[-1]
        reg_lambda_output_file = data_output_folder / f"{model_name.replace('/', '-')}_reg_lambda_{meta_data['reg_lambda']}.json"
        with open(str(reg_lambda_output_file), 'w') as file:
            json.dump(d, file, indent=4)



    # saves plots of the confidence intervals
    # plot_confidence_intervals(d, meta_data, map_lab)

