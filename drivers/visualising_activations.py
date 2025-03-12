
from torch.utils.data import DataLoader
from pathlib import Path

from utils.probe_confidence_intervals import model_setup, get_activations
from utils.steering_vector_analysis import plot_activations_PCA
from utils.preprocessing import load_txt_data

def run():
    # loads model
    print("Load model")
    model_name  = "AI-Sweden-Models/gpt-sw3-356m"
    model, tokenizer, device = model_setup(model_name)


    # loads data
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

    # set up data for plotting function
    layer = 5
    languages = ['en', 'da', 'sv', 'nb', 'is']

    activations_by_language = dict()
    for language in languages:
        acts, labels = zip(*act_ds[layer].filter_by_language(language))
        activations_by_language[language] = acts

    # plot as pca
    plot_activations_PCA(activations_by_language)
    

    



