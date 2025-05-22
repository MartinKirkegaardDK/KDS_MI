from refactor.probes import model_setup
import os
import torch
from refactor.utils.hooking import HookManager
from refactor.utils.compatibility import HookAddress
import torch.nn.functional as F
from utils.text_visualization import plot_text_viz

def main(model_name:str, model_name_temp:str, text:str,language_arg:str,layer:int):
    print("Load model") 
    model, tokenizer, device = model_setup(model_name)

    average_vectors = dict()
    path = f"average_activation_vectors/{model_name_temp}"
    for file in os.listdir(path):
        split = file.split("_")
        language = split[1]
        layer_temp = split[3]
        if int(layer_temp) == layer:
            average_vectors[language] = torch.load(path + "/" + file)
    


    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    with HookManager(model) as hook_manager:
        extracted = hook_manager.extract(HookAddress.attention_pre.layer(layer))
        
        tokenized = tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(device)

        out = model(**tokenized)
        del out


    li = []

    for token, token_activation in zip(tokenized["input_ids"][0],extracted[0]):
        token_as_word = tokenizer.decode(token).strip()
        #print(token_activation)
        current_max_similarity = 0
        most_similar_language = None
        for language, average_activation in average_vectors.items():
            similarity = cos_sim = F.cosine_similarity(average_activation.cpu().unsqueeze(0), token_activation.cpu().unsqueeze(0))

            if current_max_similarity < similarity:
                current_max_similarity = similarity
                most_similar_language = language
        li.append((current_max_similarity,most_similar_language, token_as_word))
            
    input_text = li    

    plot_text_viz(input_text,language_arg,model_name_temp,layer)