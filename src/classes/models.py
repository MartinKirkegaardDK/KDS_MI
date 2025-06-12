from torch import nn
import torch
import sklearn.metrics as metrics
import numpy as np



class ClassificationProbe(nn.Module):

    def __init__(self, in_dim, num_labs, device):
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_dim,
            out_features=num_labs,
            device=device
        )
        self.device = device
        self.accuracy = None
        self.all_preds = []
        self.all_labels = []
    
    def compute_scores(self): 
        all_preds = torch.cat(self.all_preds).cpu()
        all_labels = torch.cat(self.all_labels).cpu()
    

        self.find_used_labels()
        self.accuracy = metrics.accuracy_score(all_labels, all_preds)
        self.classification_report = metrics.classification_report(all_labels, all_preds, output_dict = True, zero_division = 1)
        
        self.class_accuracies = self.class_accuracies(self.classification_report)
    
    def find_used_labels(self):
        
        self.used_labels = set()
        for labels in self.all_labels:
      
            unique_labels = set(np.array(labels.cpu()))
            [self.used_labels.add(x) for x in unique_labels]
        

    def class_accuracies(self, classification_report):
        d = dict()
        for label in self.used_labels:
            
            d[int(label)] = self.classification_report[str(label)]["f1-score"]
        return d 


    def forward(self, x):

        return torch.softmax(self.linear(x), dim=1)