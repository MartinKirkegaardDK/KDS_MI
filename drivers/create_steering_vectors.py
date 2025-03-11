from pathlib import Path
from utils.preprocessing import load_txt_data
from utils.probe_confidence_intervals import model_setup
from utils.create_steering_vectors import get_steering_vectors
from utils.distance_plots import load_all_steering_vectors
import torch
import os

def run(model_name:str,target_language: str,complement_languages:list, run_name:str, steering_vector_path: str):
    """
    Args:
        model_name (str): _description_
        target_language (str): The language that you want to steer the model towards. Example: 'da'
        complement_languages (list): List of languages that you want to stear away from. Example: ['en','sv']
        run_name (str): This is the folder path you want to save to inside of the steering_vectors folder.
        The function automaticly saves both target, complement and combined steering vectors.
        the combined steering vector is the target - complement
        Example: 'test_run'
        steering_vector_path (str): the path for where you have saved the average activation vectors
        Example: "average_activation_vectors/gpt_sw3_356m/"

    """
    
    #all_steering_vectos = compute_all_steering_vectors(ds,languages,meta_data, tokenizer, device, model)
    all_steering_vectos = load_all_steering_vectors(steering_vector_path)
    
    target_steering_vectors, complement_steering_vectors = get_steering_vectors(all_steering_vectos, target_language, complement_languages)

    combined_vector_dict = dict()
    for layer, vectors in enumerate(zip(target_steering_vectors.values(),complement_steering_vectors.values() )):
        target_vector = vectors[0]
        complement_vector = vectors[1]
        combined = target_vector - complement_vector
        combined_vector_dict[layer] = combined
    
    if not os.path.exists(f"steering_vectors/{run_name}"):
        os.makedirs(f"steering_vectors/{run_name}")
    
    for layer, tensor in target_steering_vectors.items():
        torch.save(tensor, f"steering_vectors/{run_name}/target_steering_vector_layer_{layer}_tensor.pt")
    
    for layer, tensor in complement_steering_vectors.items():
        torch.save(tensor, f"steering_vectors/{run_name}/complement_steering_vector_layer_{layer}_tensor.pt")
    
    for layer, tensor in combined_vector_dict.items():
        torch.save(tensor, f"steering_vectors/{run_name}/combined_steering_vector_layer_{layer}_tensor.pt")
    
    

