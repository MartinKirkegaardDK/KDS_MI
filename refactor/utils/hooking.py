import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

from refactor.utils.compatibility import Hookpoints, HookAddress, ModelConfig, Device
from refactor.utils.data import ActivationDataset

import gc

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

    def _pre_extract(self, hookpoint):

        extracted = []

        def extract_hook(module, input):
            if type(input) == tuple:
                input = input[0]
            extracted.append(input.squeeze(0).detach())

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(extract_hook)
        )

        return extracted

    def _post_extract(self, hookpoint):

        extracted = []

        def extract_hook(module, input, output):
            if type(output) == tuple:
                output = output[0]
            extracted.append(output.squeeze(0).detach())

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(extract_hook)
        )

        return extracted

    def _pre_steer(self, hookpoint, steering_vector, scalar):

        def steering_hook(module, input):
            if type(input) == tuple:
                input = input[0]
            return input + steering_vector * scalar
        
        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(steering_hook)
        )

    def _post_steer(self, hookpoint, steering_vector, scalar):

        def steering_hook(module, input, output):
            if type(output) == tuple:
                output = output[0]
            return output + steering_vector * scalar
        
        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(steering_hook)
        )

    def extract(self, hook_address):

        address, placement = hook_address.split(':')

        if placement == 'pre':
            return self._pre_extract(Hookpoints.from_address(address, self.model))
        if placement == 'post':
            return self._post_extract(Hookpoints.from_address(address, self.model))
        
        raise(Exception("hook_address has to end with either ``pre'' or ``post''"))
    
    def steer(self, hook_address, steering_vector, scalar):

        address, placement = hook_address.split(':')

        if placement == 'pre':
            return self._pre_steer(Hookpoints.from_address(address, self.model), steering_vector, scalar)
        if placement == 'post':
            return self._post_steer(Hookpoints.from_address(address, self.model), steering_vector, scalar)
        
        raise(Exception("hook_address has to end with either ``pre'' or ``post''"))

        


def get_activations(
        loader: DataLoader, 
        model:AutoModelForCausalLM,
        tokenizer: AutoTokenizer, 
        hook_addresses=None,
        layers=None,
        label_map=None,
        max_batches=None,
        sampling_prob=1
        ) -> dict: 

    if not layers:
        layers = list(range(ModelConfig.hidden_layers(model)))

    if not hook_addresses:
        hook_addresses = list(HookAddress)

    if label_map == None:
        label_map = loader.dataset.label_map

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    if max_batches == None:
        max_batches = float('inf')


    activation_ds = {
        hook_address.layer(layer): ActivationDataset(label_map=label_map)
        for layer in layers
            for hook_address in hook_addresses
    }

    for ind, (text, label) in enumerate(tqdm(loader)):

        # stop early
        if ind > max_batches:
            break
        
        # extract activations
        extracted = dict()
        with HookManager(model) as hook_manager:

            for layer in layers:
                for hook_address in hook_addresses:
                    extracted[hook_address.layer(layer)] = hook_manager.extract(hook_address.layer(layer))

            tokenized = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(Device.device(model))

            out = model(**tokenized)
            del out


        # shapa data to fit in dataset class
        attn_mask = tokenized.attention_mask.flatten().detach() # # flattening [batch, pad_size, ...] to [tokens, ...]
        label = label.unsqueeze(-1).expand(-1, tokenized.attention_mask.shape[1]).flatten().detach() # [tokens]
        for layer in layers:
            for hook_address in hook_addresses:
                to_add = extracted[hook_address.layer(layer)][0].view(-1, ModelConfig.hidden_size(model)).detach() # [tokens, hidden_size]

                # add to dataset
                activation_ds[hook_address.layer(layer)].add_with_mask(to_add, label, attn_mask, sampling_prob=sampling_prob)

        # del extracted
        # del to_add
        # del attn_mask
        # del label

        # gc.collect()
        if Device.device(model) == torch.device('cuda:0'):
            torch.cuda.empty_cache()
            
    return activation_ds


