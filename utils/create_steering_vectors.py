from classes.datahandling import TextClassificationDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from utils.probe_confidence_intervals import get_activations
import torch
from collections import defaultdict


def compute_all_steering_vectors(ds: TextClassificationDataset, 
                        languages:list,
                        meta_data: dict,
                        tokenizer: AutoTokenizer,
                        device: str,
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

    Returns:
        dict: returns a dictionary with all the languages and the average activation vector across the layers
    """
    #gandhi's birthday was born in 1869
    torch.manual_seed(69)
    
    d = dict()
    for lang in languages:
        
        filtered_ds = ds.filter_by_language(lang)
        loader = DataLoader(filtered_ds, batch_size=32, shuffle=True)
        activation_ds_by_layer = get_activations(meta_data,loader, tokenizer, device, model, label_map=ds.label_map)
        #Each key has a list of averaged activations meaning that d['en'][2] is the english steering vector
        #for the 2nd layer
        d[lang] = [torch.stack(layer.predictors).mean(dim=0) for layer in activation_ds_by_layer.values()]
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
        for layer, vector in enumerate(all_steering_vectos[lang]):
            temp_d[layer].append(vector)
    #We represent it as dicts so that it is clear that each key is a layer
    complement_steering_vectors = {layer: torch.stack(value).mean(dim=0) for layer, value in temp_d.items()}
    target_steering_vectors = {i: all_steering_vectos[target_language][i] for i in range(len(all_steering_vectos[target_language]))}
    return target_steering_vectors, complement_steering_vectors