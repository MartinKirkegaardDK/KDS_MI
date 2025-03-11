from pathlib import Path
from utils.preprocessing import load_txt_data
from utils.probe_confidence_intervals import model_setup
from utils.create_steering_vectors import compute_all_steering_vectors
from utils.distance_plots import plot_distances, compute_distance_metric, load_all_steering_vectors



def run(target_language: str, steering_vector_path:str ,distance_metric: str ):
    """computes the distance plots

    Args:
        target_language (str): example: da
        steering_vector_path (str): Example: average_activation_vectors/gpt_sw3_356m/
        distance_metric (str): mahalanobis or eucled
    """

    all_steering_vectos = load_all_steering_vectors(steering_vector_path)

    distance_dict = compute_distance_metric(all_steering_vectos, target_language, distance_metric)
    
    plot_distances(distance_dict,target_language,distance_metric)