from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils.compatibility import Hookpoints, ModelConfig, Device
from utils.data import ActivationDataset

class HookManager():
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract(self, layer):
        extracted_output = []
        def extract_hook(module, input, output):
            extracted_output.append(input[0].squeeze(0).detach())

        hookpoint = Hookpoints.extraction(self.model, layer)
        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(extract_hook)
        )

        return extracted_output
    
    def steer(self, layer, steering_vector, scalar):
        def steering_hook(module, input, output):
            return output[0] + steering_vector * scalar

        hookpoint = Hookpoints.steering(self.model, layer)
        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(steering_hook)
        )



def get_activations(
        loader: DataLoader, 
        model:AutoModelForCausalLM,
        tokenizer: AutoTokenizer, 
        label_map=None,
        layers=None,
        max_batches=None
        ) -> dict: 

    if not layers:
        layers = list(range(ModelConfig.hidden_layers(model)))

    if label_map == None:
        label_map = loader.dataset.label_map

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token



    activation_ds = {
        layer: ActivationDataset(label_map=label_map)
        for layer in layers
    }

    for ind, (text, label) in enumerate(tqdm(loader)):

        # stop early
        if ind > max_batches:
            break
        
        # extract activations
        extracted = dict()
        with HookManager(model) as hook_manager:

            for layer in layers:
                extracted[layer] = hook_manager.extract(layer=layer)

            tokenized = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(Device.device(model))

            model(**tokenized)


        # shapa data to fit in dataset class
        attn_mask = tokenized.attention_mask.flatten() # # flattening [batch, pad_size, ...] to [tokens, ...]
        label = label.unsqueeze(-1).expand(-1, tokenized.attention_mask.shape[1]).flatten() # [tokens]
        for layer in layers:
            extracted[layer] = extracted[layer][0].view(-1, ModelConfig.hidden_size(model)) # [tokens, hidden_size]

        # add to appropriate dataset
        for layer in layers:
            activation_ds[layer].add_with_mask(extracted[layer], label, attn_mask)
    return activation_ds


