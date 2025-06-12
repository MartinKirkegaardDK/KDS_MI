from src.classes.datahandling import TextClassificationDataset
from torch import Generator
from torch.utils.data import random_split
from pathlib import Path


def load_txt_data(
        file_paths: dict[str, str],
        file_extension='txt') -> TextClassificationDataset:
    '''
    Loads text data from multiple language files.
    
    Args:
        file_paths (dict): Language code to file path mappings where each key represents a language code (e.g., 'en', 'da', 'sv') and each value is the corresponding file path.
        file_extention: str in {'txt', 'tml'}   
        min_sentence_length: int for min amount of tokens in sentences loaded.
    
    Returns:
        TextClassificationDataset: a dataset for each language, with labels [0;n-1] corresponding to each language
    '''

    lan_codes, file_paths = zip(*file_paths.items())

    if file_extension != 'txt':
        raise NotImplementedError('Other type extensions that "txt" have not been implemented')
    
    ds = TextClassificationDataset.from_txt(file_paths[0], lan_codes[0])

    if len(lan_codes) > 1:
        for lan_code, file_path in zip(lan_codes[1:], file_paths[1:]):
            ds.add_from_txt(file_path, lan_code)

    return ds



def filter_short_sentences(
        ds: TextClassificationDataset,
        min_sentence_length=30):
    '''
    Removes sentences from TextClassificationDataset that are under a certain length (amount of characters)

    Args:
        ds: TextClassificationDataset
        min_sentence_length: self-explanatory

    Returns:
        None (It modifies the dataset in-place)
    '''
    idx = 0
    while idx < len(ds):
        if len(ds.predictors[idx]) < min_sentence_length:
            del ds.predictors[idx]
            del ds.labels[idx]
        else:
            idx += 1

    

def split_text_data(
        ds: TextClassificationDataset,
        out_folder: str,
        split: tuple = (0.9, 0.1),
        seed: int = 42):
    '''
    Splits text data into train and test

    Args:
        ds: TextClassificationDataset
        out_folder: str the folder for the test and train directories
        split: a tuple of the form (float, float), summing to 1. (0.9, 0.1) means 90% in train and 10% in test
        seed: int random seed


    Returns:
        None (just created files)
    '''


    out_folder = Path(out_folder)
    labels = ds.label_map.keys()

    for label in labels:
        label_subset = ds.filter_by_language(label)

        train, test = random_split(
            dataset=label_subset, 
            lengths=split,
            generator=Generator().manual_seed(seed))

        train_path = out_folder / 'train' / f'{label}.txt'
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(train_path), 'w', encoding='utf-8') as f:
            for sentence, _ in train:
                f.write(sentence + '\n')

        test_path = out_folder / 'test' / f'{label}.txt'
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(test_path), 'w', encoding='utf-8') as f:
            for sentence, _ in test:
                f.write(sentence + '\n')







