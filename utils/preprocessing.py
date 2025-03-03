from classes.datahandling import TextClassificationDataset




def load_txt_data(
    file_extension='txt',
    **kwargs: str) -> TextClassificationDataset:
    '''
    Load text data from multiple language files.
    
    Args:
        **kwargs: Language code to file path mappings where each keyword argument name represents a language code (e.g., 'en', 'da', 'sv') and each value is the corresponding file path.
        
        file_extention: str in {'txt', 'tml'}
        
        min_sentence_length: int for min amount of tokens in sentences loaded.
    
    Returns:
        TextClassificationDataset: a dataset for each language, with labels [0;n-1] corresponding to each language
    '''

    lan_codes, file_paths = zip(*kwargs.items())

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

    for idx, (sentence, label) in enumerate(ds):
        if len(sentence) < min_sentence_length:
            del ds.predictors[idx]
            del ds.labels[idx]

    








