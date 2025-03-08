from torch.utils.data import Dataset, Subset
from translate.storage.tmx import tmxfile


class ParallelDataset(Dataset):
    def __init__(self, sentences: dict[str, list]):
        assert len(set([len(language) for language in sentences.values()])) == 1 

        self.sentences = sentences
        self.languages = list(self.sentences.keys())

    def __getitem__(self, index) -> tuple:
        return {language: self.sentences[language][index] for language in self.languages}

    def __len__(self) -> int:
        return len(self.sentences[self.languages[0]])
    
    @classmethod
    def from_tmx(cls, filename, lan1, lan2):
        
        with open(filename, 'rb') as file:
            tmx_file = tmxfile(
                inputfile=file, 
                sourcelanguage=lan1, 
                targetlanguage=lan2)

        lan1_sentences = []
        lan2_sentences = []
        for node in tmx_file.unit_iter():
            if node.source == '' or node.target == '': #filter empty sentences
                continue
            lan1_sentences.append(node.source)
            lan2_sentences.append(node.target)

        new_obj = cls(sentences={lan1: lan1_sentences, lan2: lan2_sentences})

        return new_obj
    

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
    def from_tmx(cls, filename, lan1, lan2):
        
        with open(filename, 'rb') as file:
            tmx_file = tmxfile(
                inputfile=file, 
                sourcelanguage=lan1, 
                targetlanguage=lan2)

        lan1_sentences = []
        lan2_sentences = []
        for node in tmx_file.unit_iter():
            if node.source == '' or node.target == '': #filter empty sentences
                continue
            lan1_sentences.append(node.source)
            lan2_sentences.append(node.target)


        lan1_sentence_pairs = []
        lan2_sentence_pairs = []

        for i in range(len(lan2_sentences) - 1):
            lan1_sentence_pairs.append((lan1_sentences[i], lan1_sentences[i + 1]))
            lan2_sentence_pairs.append((lan2_sentences[i], lan2_sentences[i + 1]))

        new_obj = cls(sentence_pairs={lan1: lan1_sentence_pairs, lan2: lan2_sentence_pairs})

        return new_obj



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


class TextClassificationDataset(ClassificationDataset):

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

    
    @classmethod
    def from_tmx(cls, filename, lan1, lan2):
        
        with open(filename, 'rb') as file:
            tmx_file = tmxfile(
                inputfile=file, 
                sourcelanguage=lan1, 
                targetlanguage=lan2)

        sentences = []
        labels = []

        for node in tmx_file.unit_iter():

            if node.source == '' or node.target == '': #filter empty sentences
                continue

            # lan1
            sentences.append(node.source)
            labels.append(0)

            # lan2
            sentences.append(node.target)
            labels.append(1)

        new_obj = cls(
            predictors=sentences,
            labels=labels
        )


        return new_obj
    
    @classmethod
    def from_tsv(cls, filename):
        sentences = []
        labels = []
        with open(filename, 'r', encoding='utf8') as file:
            for line in file.readlines():
                sentence, label = line.strip('\n').split('\t')
                sentences.append(sentence)
                labels.append(int(label))
        
        new_obj = cls(
            predictors=sentences,
            labels=labels
        )
        return new_obj