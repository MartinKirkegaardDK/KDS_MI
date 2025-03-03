import torch
from torch import nn
from torch.utils.data import Dataset
from translate.storage.tmx import tmxfile
import sklearn.metrics as metrics
import numpy as np


    

class ProbeTrainer():
    def __init__(self, input_size, num_labs, learning_rate, reg_lambda, device):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self.model = ClassificationProbe(in_dim=self.input_size, num_labs=num_labs, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.batches = 0

        self.losses = []

    def train_step(self, input_, labels, attn_mask):
        outputs = self.model(input_)
        loss = self.loss_fn(outputs, labels)
        pre_reg_loss = loss.item()
        loss += self.reg_lambda * sum(torch.norm(param, 2) for param in self.model.parameters())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.losses.append((loss.item(), pre_reg_loss))
        self.batches += 1
        return loss
    


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch.utils.data import DataLoader

    model_name = "roneneldan/TinyStories-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    try:
        hidden_size = model.config.n_embd
    except AttributeError:
        hidden_size = model.config.hidden_size

    data_loc = 'data/antibiotic/'
    ds = TextClassificationDataset.from_txt(data_loc + 'da.txt', 'da')
    ds.add_from_txt(data_loc + 'en.txt', 'en')
    ds.add_from_txt(data_loc + 'is.txt', 'is')
    ds.add_from_txt(data_loc + 'nb.txt', 'nb')
    ds.add_from_txt(data_loc + 'sv.txt', 'sv')

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    trainer = ProbeTrainer(hidden_size, 5, 0.001, 0.1, 'cpu')

    for text_batch, labels in loader:
        print(text)
        with HookManager(model) as hook_manager:
            res_stream_act = hook_manager.attach_residstream_hook(
                layer=4,
                pre_mlp=False
            )

            tokenized = [
                tokenizer(text, return_tensors='pt')
                for text in text_batch
            ]
            for text in tokenized:
                model.forward(**text)

        loss = trainer.train_step(torch.Tensor(res_stream_act), torch.Tensor(labels))
        print(loss)





