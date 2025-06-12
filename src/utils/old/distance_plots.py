import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean, mahalanobis, cosine
from collections import defaultdict
import os
import torch

def load_all_steering_vectors(path):
    temp = []
    all_steering_vectos = defaultdict(list)
    for file in os.listdir(path):
        li = file.split("_")
        language = li[1]
        layer = li[3]
        steering_vector = torch.load(str(path + file))
        temp.append((language,layer, steering_vector))
    data_sorted = sorted(temp, key=lambda x: (x[0], int(x[1])))
    for language, layer, tensor in data_sorted:
        all_steering_vectos[language].append(tensor)
    return all_steering_vectos

def plot_distances(d: dict,target_language:str,type_distance:str, ax, title='', labels=False):
    """_summary_

    Args:
        d (dict): d[lang][layer distances] each key is a language that contains a list with the layer distances
        target_language (str): this could be da, meaning that it is da's distance to all the other languages
        type_distance (str): mahalanobis, euclidean ect.
        name (_type_, optional): name of the figure you want to save as. Has build in name generator. Defaults to None.
    """
    
    # Create the plot with similar styling
    
    

    # Use a distinct color palette
    colors = plt.cm.Set1(range(5))
    
    color_map = {
        'en': 1,
        'is': 2,
        'sv': 3,
        'nb': 4
    }

    # Slightly offset x positions for different labels to avoid direct overlap
    offset_step = 0.1

    for idx, key in enumerate(color_map):
        values = d[key]
        x_positions = np.arange(len(values)) + (idx - (len(d) - 1) / 2) * offset_step

        # Plot the line with markers
        if labels:
            ax.plot(x_positions, values, linewidth=1, color=colors[color_map[key]], label=key)
        else:
            ax.plot(x_positions, values, linewidth=1, color=colors[color_map[key]])
            
    # Improve the overall appearance
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel(f'{type_distance.capitalize()} distance to {target_language}')

    # Customize x-ticks to match indices
    #ax.set_xticks(np.arange(max(len(v) for v in d.values())))

    
    #plt.tight_layout()

    #plt.savefig(f'results/activation_vector_distances/{name}_{type_distance}_{target_language}.png', bbox_inches='tight')

    #plt.show()
    
    
def compute_distance_metric(all_steering_vectos:dict, target_language:str , distance_metric: str) -> dict:
    """computes the distance metric for a target language

    Args:
        target_language (str): the language which you want to compute all the distances against
        all_steering_vectos (dict): a dict containing all the computed steering vectors
        distance_metric (str): some distance metric, euclidean, mahalanobis ect.

    Returns:
        dict: a dict containing all the distances and languages
    """
    d = defaultdict(list)
    if distance_metric == "euclidean":
        for language in all_steering_vectos.keys():
            if language == target_language:
                continue
            for lang_vector, da_vector in zip(all_steering_vectos[language],all_steering_vectos[target_language]):
                dist = euclidean(lang_vector.cpu(), da_vector.cpu())
                d[language].append(dist)
                
    elif distance_metric == "cosine":
        for language in all_steering_vectos.keys():
            if language == target_language:
                continue
            for lang_vector, da_vector in zip(all_steering_vectos[language],all_steering_vectos[target_language]):
                dist = cosine(lang_vector.cpu(), da_vector.cpu())
                d[language].append(dist)
                
    elif distance_metric == "mahalanobis":
        for language in all_steering_vectos.keys():
            if language == target_language:
                continue
            for lang_vector, da_vector in zip(all_steering_vectos[language],all_steering_vectos[target_language]):
                V = np.cov(np.array([lang_vector.cpu(), da_vector.cpu()]).T)
                #Maybe we need to find a better covariance matrix. Maybe it needs to be for all vectors???
                IV = np.linalg.pinv(V) #pseudoinverse due to the matrix V being singular
                dist = mahalanobis(lang_vector.cpu(), da_vector.cpu(),IV)
                d[language].append(dist)
    else:
        raise Exception("The distance metric provided to the function is not allowed. Currently only euclidean, and mahalanobis are allowed")
    
    return d