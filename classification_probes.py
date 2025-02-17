import torch
from torch import nn
from torch.utils.data import Dataset
from translate.storage.tmx import tmxfile

class TextClassificationDataset(Dataset):

    def __init__(self, sentences, labels):
        super().__init__()

        assert len(sentences) == len(labels)
        
        self.sentences = sentences
        self.labels = labels


    def __getitem__(self, index) -> tuple:
        return (self.sentences[index], self.labels[index]) 
    

    def __len__(self) -> int:
        return len(self.sentences)
    
    @classmethod
    def from_txt(cls, filename, language):
        sentences = []
        labels = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                sentences.append(line.strip())
                labels.append(language)

        new_obj = cls(
            sentences=sentences,
            labels=labels
        )

        return new_obj
    
    def add_from_txt(self, filename, language):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                self.sentences.append(line.strip())
                self.labels.append(language)

    
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
            sentences=sentences,
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
            sentences=sentences,
            labels=labels
        )
        return new_obj



class HookManager():
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def attach_residstream_hook(self, layer, pre_mlp=False, pythia=False):
        if pre_mlp:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.mlp'
            else:
                hookpoint = f'transformer.h.{layer}.mlp'
        else:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.attention'
            else:
                hookpoint = f'transformer.h.{layer}.attn'
        
        extracted_output = []
        def residstream_hook(module, input, output):
            extracted_output.append(input[0].squeeze(0).detach())

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(residstream_hook)
        )

        return extracted_output
    
    def attach_resid_stream_steer_hook(self, layer, steering_vector, scalar, pre_mlp=False, pythia=False):
        if pre_mlp:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.mlp'
            else:
                hookpoint = f'transformer.h.{layer}.mlp'
        else:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.attention'
            else:
                hookpoint = f'transformer.h.{layer}.attn'

        def steering_hook(module, input):
            activation = input[0]

            steering_norm = steering_vector / torch.norm(steering_vector)
            
            projection_magnitudes = (activation @ steering_norm).unsqueeze(-1)
            
            steering_norm_ = steering_norm.view(1, 1, -1)

            projections = projection_magnitudes * steering_norm_

            modified = activation + scalar * projections

            act_norm = torch.norm(activation, dim=2).unsqueeze(-1)
            modified_norm = torch.norm(activation, dim=2).unsqueeze(-1)

            modified = modified * (act_norm / modified_norm)

            return (modified,) + input[1:] if len(input) > 1 else (modified,)
        
        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(steering_hook)
        )
        
        

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ClassificationProbe(nn.Module):

    def __init__(self, in_dim, num_labs, device):
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_dim,
            out_features=num_labs,
            device=device
        )

    def forward(self, x):

        return torch.softmax(self.linear(x), dim=1)
    

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





