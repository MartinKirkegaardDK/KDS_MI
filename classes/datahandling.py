from torch.utils.data import Dataset, Subset
from translate.storage.tmx import tmxfile


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