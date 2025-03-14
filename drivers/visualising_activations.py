
from torch.utils.data import DataLoader
from pathlib import Path
from matplotlib import pyplot as plt
from math import sqrt

from utils.probe_confidence_intervals import model_setup, get_activations
from utils.plotting import plot_activations_PCA
from utils.preprocessing import load_txt_data

def run(model_name):
    
    # loads model
    model, tokenizer, device = model_setup(model_name)
    hidden_layers = model.config.num_hidden_layers


    # loads data
    languages = ['en', 'da', 'sv', 'nb', 'is']
    raw_data_folder = Path('data/antibiotic/')
    print("Load data")
    ds = load_txt_data(
        file_paths={
            'da': raw_data_folder / 'da.txt',
            'en': raw_data_folder / 'en.txt',
            'sv': raw_data_folder / 'sv.txt',
            'nb': raw_data_folder / 'nb.txt',
            'is': raw_data_folder / 'is.txt'
        },
        file_extension='txt'
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)


    # extracts activation from forward passes on data    
    act_ds = get_activations(
        meta_data={'hidden_layers': model.config.num_hidden_layers,
                   'hidden_size': model.config.hidden_size},
        loader=loader,
        tokenizer=tokenizer,
        model=model,
        device=device
    )


    # makes figure
    ncols = int(sqrt(hidden_layers)) + int((hidden_layers % hidden_layers) != 0)
    nrows = hidden_layers // ncols + int(hidden_layers % ncols != 0)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,20))
    axs = axs.flatten()


    for layer in range(hidden_layers):

        # set up data for plotting function
        activations_by_language = dict()
        for language in languages:
            acts, labels = zip(*act_ds[layer].filter_by_language(language))
            activations_by_language[language] = acts

        # plot as pca
        plot_activations_PCA(activations_by_language, layer=layer, ax=axs[layer])
    

    fig.tight_layout()
    



