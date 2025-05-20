from torch.utils.data import DataLoader
from refactor.utils.data import FilePaths, load_antibiotic_data
from refactor.utils.hooking import get_activations as get_activations_new
from refactor.utils.compatibility import ModelConfig
from refactor.probes import model_setup
from utils.probe_confidence_intervals import bootstrap
from collections import defaultdict
from pathlib import Path
import numpy as np
import json


def idk(model_name,model_name_temp, reg_lambdas):
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
    meta_data["hidden_size"] = ModelConfig.hidden_size(model)
    meta_data["hidden_layers"] = ModelConfig.hidden_layers(model)
    meta_data["model_name"] = model_name.split("/")[0]
    meta_data["learning_rate"] = 0.001
    meta_data["amount_epochs"] = 1


    # extracts activation from forward passes on data
    # We use hooks to extract the different layer activations that will be used to train our probes

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
    
    #Here we get the different positions that we are testing
    positions = []
    for index, (key, val) in enumerate(activations.items()):
        if index == 6: break
        positions.append(key.replace("layer.0.",""))
    
    
    d = {}
    for pos in positions:
        acts_ds_by_layer = {}
        for layer in range(meta_data["hidden_layers"]):
            pos_key = f"layer.{layer}.{pos}"
            acts_ds_by_layer[layer] = activations[pos_key]
        d[pos] = acts_ds_by_layer
    
    #We extract the amount of labels. 
    #We just do this for a single position as all of them shares the same labels
    s = set()
    for i in range(meta_data["hidden_layers"]):
        unique_labels = set(np.array(acts_ds_by_layer[i].labels))
        [s.add(x) for x in unique_labels]
    number_labels = len(s)
    meta_data["number_labels"] = number_labels



    data_output_folder = Path('results/data/probe_confidence_intervals')

    for pos in positions:
        
        acts_ds_by_layer = d[pos]
        print(acts_ds_by_layer, pos, "hi")
        for reg_lambda in reg_lambdas:
            #print()
            meta_data['reg_lambda'] = reg_lambda
            boot = bootstrap(10, meta_data, acts_ds_by_layer, device)
            map_lab = ds.map_label

            
            d_temp = defaultdict(list)
            for run in boot:
                for layer in run.keys():
                    class_accuracies = run[layer].class_accuracies
                    d_temp[layer].append(class_accuracies)



            # saves data used in plots
            
            d_temp['map_label'] = map_lab

            reg_lambda_output_file = data_output_folder / f"{model_name_temp}_{pos}_reg_lambda_{meta_data['reg_lambda']}.json"
            with open(str(reg_lambda_output_file), 'w') as file:
                json.dump(d_temp, file, indent=4)
