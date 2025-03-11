from pathlib import Path
from utils.preprocessing import load_txt_data
from utils.probe_confidence_intervals import model_setup
from utils.create_steering_vectors import compute_all_steering_vectors
import os
import torch

def run(languages:list[str],model_name:str,run_name:str):
    """This function creates and saves the average_activation_vectors. This variable may also be called
    all_steering_vectos
    All the vectors are saved under average_activation_vectors/run_name

    Args:
        languages (list[str]): the languages that you want to extract
        model_name (str): the model you want to use
        run_name (str): the output name 
    """

    model, tokenizer, device = model_setup(model_name)

    raw_data_folder = Path('data/antibiotic/')
    print("Load data")

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
    

    if not os.path.exists(f"average_activation_vectors/{run_name}"):
        os.makedirs(f"average_activation_vectors/{run_name}")

    for language, list_of_tensors in all_steering_vectos.items():
        for layer, tensor in enumerate(list_of_tensors):
            torch.save(tensor, f"average_activation_vectors/{run_name}/language_{language}_layer_{layer}_tensor.pt")
