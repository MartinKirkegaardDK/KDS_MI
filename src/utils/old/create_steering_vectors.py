from src.classes.datahandling import TextClassificationDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from src.utils.new.hooking import get_activations
import torch
from collections import defaultdict
from pathlib import Path

def compute_all_steering_vectors(ds: TextClassificationDataset, 
                        languages:list,
                        tokenizer: AutoTokenizer,
                        model: AutoModelForCausalLM) -> dict:
    """This function computes the average activations for all the languages.
    The target steering vector for danish is simply the average activation for danish.

    Args:
        ds (TextClassificationDataset): the dataset class. We use it to filter languages
        languages (list): the list of all the languages you want to use
        meta_data (dict): meta_data containing hidden_layers and hidden_size
        tokenizer (AutoTokenizer): tokenizer for the model
        device (str): the device as a str, example: cpu
        model (AutoModelForCausalLM): huggingface model
        model_name (str): The name of the model
    Returns:
        dict: returns a dictionary with all the languages and the average activation vector across the layers
    """
    #gandhi's birthday was born in 1869
    torch.manual_seed(69)
    
    # saved_path_raw_activations = "raw_activations/"
    
    # Path(f"{saved_path_raw_activations}/{model_name}").mkdir(parents=True, exist_ok=True)

    
    d = dict()
    for lang in languages:
        
        filtered_ds = ds.filter_by_language(lang)
        # print(filtered_ds)
        loader = DataLoader(filtered_ds, batch_size=32, shuffle=True)
        activation_ds = get_activations(
            loader=loader, 
            model=model, 
            tokenizer=tokenizer,
            label_map=ds.label_map,
            max_batches=20,
            sampling_prob=0.1
        )
        
        # for hook_address_, ds_ in activation_ds.items():
        #     hook_address = hook_address_.replace(':', '_')
        #     torch.save(ds_, f'{saved_path_raw_activations}/{model_name}/layer_{layer}_language_{lang}_tensors.pt')

        #Each key has a list of averaged activations meaning that d['en'][2] is the english steering vector
        #for the 2nd layer
        d[lang] = {hook_address: torch.stack(activations.predictors).mean(dim=0) for hook_address, activations in activation_ds.items()}
    return d
    

def get_steering_vectors(all_steering_vectos:dict,target_language: str, complement_languages:list) -> list[dict,dict]:
    """This function transforms all the steering vectors into two, the target_steering_vectors and complement_steering_vectors

    Args:
        all_steering_vectos (dict): all the steering vectors which is going to be split up
        target_language (str): the language, ect da, en
        complement_languages (list): the complement languages, meaning the languages minus your target language

    Returns:
        list[dict,dict]: returns the target_steering_vectors and complement_steering_vectors
    """
    temp_d = defaultdict(list)

    for lang in complement_languages:
        for hook_address, vector in all_steering_vectos[lang].items():
            temp_d[hook_address].append(vector)
    #We represent it as dicts so that it is clear that each key is a layer
    complement_steering_vectors = {layer: torch.stack(value).mean(dim=0) for layer, value in temp_d.items()}
    # target_steering_vectors = {i: all_steering_vectos[target_language][i] for i in range(len(all_steering_vectos[target_language]))}

    target_steering_vectors = all_steering_vectos[target_language]
    return target_steering_vectors, complement_steering_vectors