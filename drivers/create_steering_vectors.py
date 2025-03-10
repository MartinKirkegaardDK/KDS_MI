from pathlib import Path
from utils.preprocessing import load_txt_data
from utils.probe_confidence_intervals import model_setup
from utils.create_steering_vectors import compute_all_steering_vectors, get_steering_vectors
import torch
import os

def run(model_name:str,target_language: str,complement_languages:list, run_name:str ):
    """
    Args:
        model_name (str): _description_
        target_language (str): The language that you want to steer the model towards. Example: 'da'
        complement_languages (list): List of languages that you want to stear away from. Example: ['en','sv']
        run_name (str): This is the folder path you want to save to inside of the steering_vectors folder.
        The function automaticly saves both target, complement and combined steering vectors.
        Example: 'test_run'
    """
    print("Load model")

    model, tokenizer, device = model_setup(model_name)

    raw_data_folder = Path('data/preprocessed/train')
    print("Load data")
    
    languages = complement_languages + [target_language]
    file_paths = {lang: raw_data_folder / f'{lang}.txt' for lang in languages }
    
    ds = load_txt_data(
        file_paths= file_paths,
        file_extension='txt'
    )
    
    meta_data = {}
    meta_data["hidden_layers"] = model.config.num_hidden_layers

    try:
        meta_data["hidden_size"] = model.config.n_embd
    except AttributeError:
        meta_data["hidden_size"] = model.config.hidden_size
    
    
    all_steering_vectos = compute_all_steering_vectors(ds,languages,meta_data, tokenizer, device, model)
    
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
    
    

