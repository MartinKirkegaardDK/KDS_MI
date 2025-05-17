import torch
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass

antibiotic_folder = lambda lan: f'data/antibiotic/{lan}.txt'

class FilePaths:    
    antibiotic = {
        'da': antibiotic_folder('da'),
        'en': antibiotic_folder('en'),
        'is': antibiotic_folder('is'),
        'sv': antibiotic_folder('sv'),
        'nb': antibiotic_folder('nb')
    }


class ClassificationDataset(Dataset):

    def __init__(self, predictors, labels, label_map=None):
        super().__init__()

        assert len(predictors) == len(labels)
        
        self.predictors = predictors
        self.labels = labels

        if label_map == None:
            self.label_map = dict()
        else:
            self.label_map = label_map

    @property
    def map_label(self):
        return {value: key for key, value in self.label_map.items()}

    def __getitem__(self, index) -> tuple:
        return (self.predictors[index], self.labels[index])

    def __len__(self) -> int:
        return len(self.predictors)
    

    def filter_by_language(self, language):

        assert len(self.label_map) > 0

        indices = [
            idx 
            for idx, label 
                in enumerate(self.labels)
            if self.label_map[language] == label 
        ]

        return Subset(self, indices)



class AntibioticDataset(ClassificationDataset):

    def __init__(self, sentences, labels):
        super().__init__(
            predictors=sentences,
            labels=labels
        )

    @classmethod
    def from_txt(cls, filename, language):
        sentences = []
        labels = []

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                sentences.append(line.strip())
                labels.append(0)

        new_obj = cls(
            sentences=sentences,
            labels=labels
        )

        new_obj.label_map[language] = 0

        return new_obj
    
    
    def add_from_txt(self, filename, language):

        if language not in self.label_map:
            self.label_map[language] = len(self.label_map)

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                self.predictors.append(line.strip())
                self.labels.append(self.label_map[language])


class ActivationDataset(ClassificationDataset):

    def __init__(self, label_map=None):
        super().__init__(
            predictors=[],
            labels=[],
            label_map=label_map
        )

    def add_with_mask(self, acts, labels, masks):
        for act, label, mask in zip(acts, labels, masks):
            if mask:
                self.predictors.append(act)
                self.labels.append(label)


    def filter_by_language(self, language, return_tensors=False):

        if not return_tensors:
            return super().filter_by_language(language)   

        acts, labels = zip(*super().filter_by_language(language))
        return torch.stack(acts, dim=0).detach()
    




def load_antibiotic_data(
        file_paths: dict[str, str],
        file_extension='txt') -> AntibioticDataset:
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
    
    ds = AntibioticDataset.from_txt(file_paths[0], lan_codes[0])

    if len(lan_codes) > 1:
        for lan_code, file_path in zip(lan_codes[1:], file_paths[1:]):
            ds.add_from_txt(file_path, lan_code)

    return ds