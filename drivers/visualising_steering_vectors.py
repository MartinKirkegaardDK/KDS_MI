import torch

from utils.steering_vector_analysis import plot_PCA

def run(steering_vector_paths_by_language: dict[str, str]):

    # loads steering vector tensors
    steering_vectors_by_language = {
        language: torch.load(steering_vector_paths_by_language[language], map_location='cpu')
        for language in steering_vector_paths_by_language.keys()
    }

    # plots steering vector tensors
    plot_PCA(steering_vectors=steering_vectors_by_language)
    
