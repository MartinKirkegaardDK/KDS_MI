from fasttext.FastText import _FastText
from classes.datahandling import ParallelNSPDataset
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from utils.steering import generate_with_steering
import os




def gen_outputs(bible_data:ParallelNSPDataset, 
                language_1:str,
                language_2:str, 
                bible_index:int, 
                layer:int,
                steering_vector:torch.Tensor,
                steering_lambda:int,
                model:AutoModelForCausalLM,
                tokenizer:AutoTokenizer) -> tuple:
    """Inserts a steering vector and shifts the model towards that direction. 
    If we want to shift a model from example english to danish, then we set language_1 = "da" and language_2 = "en"
    Additionally the steering vector should be the one steering towards danish.

    Args:
        bible_data (ParallelNSPDataset): dataset with bible data
        language_1 (str): the language you want to steer towards
        language_2 (str): the language you steer away from
        bible_index (int): index of a given verse in the bible
        layer (int): layer of the model where you want to insert the steering vector
        steering_vector (torch.Tensor): the steering vector
        steering_lambda (int): the strenght of the steering vector
        model (AutoModelForCausalLM): the model you want to use
        tokenizer (AutoTokenizer): The tokenizer used with the model

    Returns:
        tuple: _description_
    """


    language_1_prompt = bible_data[bible_index][language_1][0].lower()
    language_1_true_bible_verse = bible_data[bible_index][language_1][1]
    
    language_2_prompt = bible_data[bible_index][language_2][0].lower()
    language_2_true_bible_verse = bible_data[bible_index][language_2][1]
    
    input_ids = tokenizer(language_1_prompt, return_tensors="pt")["input_ids"]
    generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=30, do_sample=True)[0]
    language_1_predicted_bible_verse = tokenizer.decode(generated_token_ids)[len(language_1_prompt):]
    
    language_2_predicted_bible_verse = generate_with_steering(model,tokenizer,layer,language_2_prompt,steering_vector[layer], steering_lambda= steering_lambda)
    language_2_predicted_bible_verse = language_2_predicted_bible_verse[0][len(language_2_prompt):]
    
    return language_1_predicted_bible_verse, language_2_predicted_bible_verse, language_1_true_bible_verse,language_2_true_bible_verse

def load_targeted_steering_vectors(steering_vector_path: str,device: str) -> tuple[dict,dict,dict]:
    """loads steering vectors that are targeted towards a language. 
    it returns the target, complement and combined, with combined = target - complement

    Args:
        steering_vector_path (str): some path

    Returns:
        tuple[dict,dict,dict]: target, complement, combined
    """
    combined = dict()
    complement = dict()
    target = dict()
    for vector in os.listdir(steering_vector_path):
        type = vector.split("_")[0]
        layer = vector.split("_")[4]
        if type == "combined":
            combined[int(layer)] = torch.load(str(steering_vector_path +vector),map_location=torch.device(device))
        elif type == "complement":
            complement[int(layer)] = torch.load(str(steering_vector_path + vector),map_location=torch.device(device))
        elif type == "target":
            target[int(layer)] = torch.load(str(steering_vector_path +vector),map_location=torch.device(device))
    return target, complement, combined

