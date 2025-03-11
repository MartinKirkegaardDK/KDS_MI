import torch
from pathlib import Path

from utils.probe_confidence_intervals import model_setup
from utils.preprocessing import load_txt_data
from utils.steering_vector_analysis import plot_loss_for_steering_vectors

from classes.datahandling import ParallelNSPDataset

def run(steering_vector_folder, model_name):

    # loads model
    model, tokenizer, device = model_setup(model_name)

    # loads data
    bible_path = Path('data/bible-da-en.tmx')
    ds = ParallelNSPDataset.from_tmx(
        str(bible_path),
        lan1='da',
        lan2='en'
    )

    # loads steering vectors by layer
    steering_vector_folder = Path(steering_vector_folder)
    num_layers = model.config.num_hidden_layers
    steering_vectors_by_layer = {
        layer: torch.load(steering_vector_folder / f'combined_steering_vector_layer_{layer}_tensor.pt', map_location='cpu')
        for layer in range(num_layers)
    }

    # plots losses on bible data
    plot_loss_for_steering_vectors(
        model,
        tokenizer,
        ds,
        steering_vectors_by_layer,
        lan1='en',
        lan2='da',
        amount_datapoints=100
    )