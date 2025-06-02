import torch
from torch.utils.data import Dataset, Subset
import random
import xml.etree.ElementTree as ET

from markus.utils.compatibility import FilePaths, Device



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

    def add_with_mask(self, acts, labels, masks, sampling_prob=1):
        for act, label, mask in zip(acts, labels, masks):
            if mask:
                if sampling_prob == 1 or random.random() < sampling_prob:
                    self.predictors.append(act.cpu())
                    self.labels.append(label.cpu())

        # del act, acts, label, labels, mask, masks
        # gc.collect()



    def filter_by_language(self, language, return_tensors=False):

        if not return_tensors:
            return super().filter_by_language(language)   

        acts, labels = zip(*super().filter_by_language(language))
        return torch.stack(acts, dim=0).detach()
    


class ParallelNSPDataset(Dataset):
    def __init__(self, sentence_pairs: dict[str, list[tuple]]):
        assert len(set([len(language) for language in sentence_pairs.values()])) == 1 

        self.sentence_pairs = sentence_pairs
        self.languages = list(self.sentence_pairs.keys())

    def __getitem__(self, index) -> tuple:
        return {language: self.sentence_pairs[language][index] for language in self.languages}

    def __len__(self) -> int:
        return len(self.sentence_pairs[self.languages[0]])

    @classmethod
    def from_xml(cls, file_paths):

        lan_codes = list(file_paths.keys())
        xmls = {lan_code: ET.parse(file_paths[lan_code]) for lan_code in lan_codes}


        verse_ids = {lan_code: [] for lan_code in lan_codes}
        verse_texts = {lan_code: {} for lan_code in lan_codes}

        for lan_code in lan_codes:
            root = xmls[lan_code].getroot()
            verses = root.findall(".//seg[@type='verse']")
            for verse in verses:
                verse_id = verse.get('id')
                verse_ids[lan_code].append(verse_id)
                verse_texts[lan_code][verse_id] = verse.text.strip()

        # just to be sure
        lan1, lan2 = lan_codes
        lan2_verses = set(verse_ids[lan_codes[1]])
        common_verses = [verse_id for verse_id in verse_ids[lan1] if verse_id in lan2_verses]

        sentences = {lan_code: [verse_texts[lan_code][id_] for id_ in common_verses] for lan_code in lan_codes}

        lan1_sentence_pairs = []
        lan2_sentence_pairs = []

        for i in range(len(common_verses) - 1):
            lan1_sentence_pairs.append((sentences[lan1][i], sentences[lan1][i + 1]))
            lan2_sentence_pairs.append((sentences[lan2][i], sentences[lan2][i + 1]))

        new_obj = cls(sentence_pairs={lan1: lan1_sentence_pairs, lan2: lan2_sentence_pairs})

        return new_obj
    


def load_antibiotic_data() -> AntibioticDataset:
    '''
    Loads text data from multiple language files.
    
    Returns:
        TextClassificationDataset: a dataset for each language, with labels [0;n-1] corresponding to each language
    '''

    lan_codes, file_paths = zip(*FilePaths.antibiotic.items())
    
    ds = AntibioticDataset.from_txt(file_paths[0], lan_codes[0])

    if len(lan_codes) > 1:
        for lan_code, file_path in zip(lan_codes[1:], file_paths[1:]):
            ds.add_from_txt(file_path, lan_code)

    return ds



def load_bible_data(lan1, lan2):

    ds = ParallelNSPDataset.from_xml(
        file_paths={
            lan1: FilePaths.bible[lan1],
            lan2: FilePaths.bible[lan2]
        }
    )

    return ds


def load_steering_vector(lan, hook_address, model):
    
    return torch.load(FilePaths.steering_vectors(lan, hook_address, model), map_location=Device.device(model))
