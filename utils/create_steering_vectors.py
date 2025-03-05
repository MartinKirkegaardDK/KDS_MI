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
    d = dict()
    for lang in languages:
        
        filtered_ds = ds.filter_by_language(lang)
        loader = DataLoader(filtered_ds, batch_size=32, shuffle=True)
        activation_ds_by_layer = get_activations(meta_data,loader, tokenizer, device, model)
        #Each key has a list of averaged activations meaning that d['en'][2] is the english steering vector
        #for the 2nd layer
        d[lang] = [torch.stack(layer.predictors).mean(dim=0) for layer in activation_ds_by_layer.values()]
    return d
    

def get_steering_vectors(all_steering_vectos:dict,target_language: str, complement_languages:list) -> list[dict,dict]:
    temp_d = defaultdict(list)

    for lang in complement_languages:
        for layer, vector in enumerate(all_steering_vectos[lang]):
            temp_d[layer].append(vector)
    #We represent it as dicts so that it is clear that each key is a layer
    complement_steering_vectors = {layer: torch.stack(value).mean(dim=0) for layer, value in temp_d.items()}
    target_steering_vectors = {i: all_steering_vectos[target_language][i] for i in range(len(all_steering_vectos[target_language]))}
    return target_steering_vectors, complement_steering_vectors