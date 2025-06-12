from src.utils.old.distance_plots import plot_distances, compute_distance_metric, load_all_steering_vectors
from matplotlib import pyplot as plt


def run(model_name_temp:str, target_language: str, steering_vector_path:str ,distance_metric: str):
    """computes the distance plots

    Args:
        target_language (str): example: da
        steering_vector_path (str): Example: average_activation_vectors/gpt_sw3_356m/
        distance_metric (str): mahalanobis or eucled
    """
    

    all_steering_vectos = load_all_steering_vectors(steering_vector_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs = axs.flatten()
   

    metric = 'cosine'
    distance_dict = compute_distance_metric(all_steering_vectos, target_language, metric)
    plot_distances(distance_dict,target_language,metric,ax=axs[0], title=f'{metric.capitalize()} distance', labels=True)
    
    metric = 'euclidean'
    distance_dict = compute_distance_metric(all_steering_vectos, target_language, metric)
    plot_distances(distance_dict,target_language,metric,ax=axs[1], title=f'{metric.capitalize()} distance')
    
    #metric = 'mahalanobis'
    #distance_dict = compute_distance_metric(all_steering_vectos, target_language, metric)
    #plot_distances(distance_dict,target_language,metric,ax=axs[2], title=f'{metric.capitalize()} distance')
    
    fig.legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncol=4, title='language')
    fig.tight_layout()
    fig.savefig(f"results/figures/activation_distances/{model_name_temp}_{target_language}.png",bbox_inches = "tight",dpi = 300)
    