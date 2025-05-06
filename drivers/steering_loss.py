import torch
from pathlib import Path
import matplotlib.pyplot as plt

from utils.probe_confidence_intervals import model_setup
from utils.preprocessing import load_txt_data
from utils.steering_vector_analysis import plot_loss_for_steering_vectors

from classes.datahandling import ParallelNSPDataset

def run(
        steering_vector_folder, 
        model_name,
        lan1,
        lan2
    ):
    """
    Plots the steering loss and saves it at: results/steering_loss/*model_name*
    """
    saved_path = "results/steering_loss"

    # loads model
    model, tokenizer, device = model_setup(model_name)

    # loads data
    bible_path = Path('data/bible')
    ds = ParallelNSPDataset.from_xml(
        str(bible_path),
        lan1=lan1,
        lan2=lan2
    )

    steering_vector_folder = Path(steering_vector_folder)
    num_layers = model.config.num_hidden_layers
    steering_vectors_by_layer = {
        layer: torch.load(steering_vector_folder / f'combined_steering_vector_layer_{layer}_tensor.pt', map_location='cpu')
        for layer in range(num_layers)
    }

    steering_lambdas = [1, 2, 5, 10, 15]
    Path(saved_path).mkdir(parents=True, exist_ok=True)

    if "download" in model_name:
        model_name = model_name.split("/")[-1]

    for steering_lambda in steering_lambdas:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))  # One figure per lambda
        plot_loss_for_steering_vectors(
            model,
            tokenizer,
            ds,
            steering_vectors_by_layer,
            steering_lambda,
            lan1=lan1,
            lan2=lan2,
            amount_datapoints=100,
            ax=ax
        )
        fig.tight_layout()
        fig.savefig(f"{saved_path}/{model_name}_lambda_{steering_lambda}.png")
        plt.close(fig)  # Important to prevent memory issues in long runs
