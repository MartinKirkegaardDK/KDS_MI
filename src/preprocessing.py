from pathlib import Path
from src.utils.old.preprocessing import *

def run():
    raw_data_folder = Path('data/antibiotic/')

    # loads text data for all the languages
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

    # removs the sentences which are too short
    # (we found that for out particular data, there was a lot of short repeated sentences)
    filter_short_sentences(ds, min_sentence_length=40)
    

    # splits into test and train and saves them in data/preprocessed
    out_folder = Path('data/preprocessed')
    split_text_data(
        ds, 
        out_folder=out_folder,
        split=(0.9, 0.1),
        seed=42
    )
