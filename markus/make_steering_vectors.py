import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from collections import defaultdict

from markus.utils.hooking import get_activations
from markus.utils.compatibility import ModelName, FilePaths
from markus.utils.data import load_antibiotic_data


def compute_all_steering_vectors(ds, 
                        languages:list,
                        tokenizer: AutoTokenizer,
                        model: AutoModelForCausalLM) -> dict:
    """This function computes the average activations for all the languages.
    The target steering vector for danish is simply the average activation for danish.

    Args:
        ds (TextClassificationDataset): the dataset class. We use it to filter languages
        languages (list): the list of all the languages you want to use
        tokenizer (AutoTokenizer): tokenizer for the model
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
        activation_ds = get_activations(
            loader=loader, 
            model=model,
            tokenizer=tokenizer,
            label_map=ds.label_map,
            max_batches=5,
            sampling_prob=0.1
        )
        
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



def main(model_url:str,target_language: str,complement_languages:list, device, out_folder):
    """
    The function automaticly saves both target, complement and combined steering vectors.
    the combined steering vector is the target - complement
    Args:
        model_name (str): _description_
        target_language (str): The language that you want to steer the model towards. Example: 'da'
        complement_languages (list): List of languages that you want to stear away from. Example: ['en','sv']
        run_name (str): This is the folder path you want to save to inside of the steering_vectors folder.
        Example: 'test_run'

    """

    model = AutoModelForCausalLM.from_pretrained(model_url).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    model_name = ModelName.name(model)

    ds = load_antibiotic_data()

    languages = list(FilePaths.antibiotic.keys())
    
    all_steering_vectos = compute_all_steering_vectors(ds, languages, tokenizer, model)
    # steering_vector_path =f"average_activation_vectors/{model_name}/"
    # all_steering_vectos = load_all_steering_vectors(steering_vector_path)
    
    target_steering_vectors, complement_steering_vectors = get_steering_vectors(all_steering_vectos, target_language, complement_languages)

    combined_vector_dict = dict()
    for hook_address in target_steering_vectors:
        combined = target_steering_vectors[hook_address] - complement_steering_vectors[hook_address]
        combined_vector_dict[hook_address] = combined





    folder = f"markus/{out_folder}/{model_name}/{target_language}"

    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/complement", exist_ok=True)
    os.makedirs(f"{folder}/target", exist_ok=True)
    os.makedirs(f"{folder}/combined", exist_ok=True)


    for hook_address, tensor in target_steering_vectors.items():   
        torch.save(tensor, f"{folder}/target/{hook_address}.pt")
    
    for hook_address, tensor in complement_steering_vectors.items():
        torch.save(tensor, f"{folder}/complement/{hook_address}.pt")
    
    for hook_address, tensor in combined_vector_dict.items():
        torch.save(tensor, f"{folder}/combined/{hook_address}.pt")