from transformers import GPTNeoXForCausalLM, GPT2LMHeadModel
from enum import Enum

class HookAddress(Enum):
    layernorm_1_pre = 'layernorm_1:pre'
    attention_pre = 'attention:pre'
    attention_post = 'attention:post'
    layernorm_2_pre = 'layernorm_2:pre'
    mlp_pre = 'mlp:pre'
    mlp_post = 'mlp:post'

    def layer(self, layer):
        return f'layer.{layer}.{self.value}'


class Hookpoints:

    @staticmethod
    def layernorm_1(model):
        hookpoints = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.input_layernorm',
            GPT2LMHeadModel: lambda layer: f'transformer.h.{layer}.ln_1'
        }
        return hookpoints[type(model)]
    
    @staticmethod
    def layernorm_2(model):
        hookpoints = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.post_attention_layernorm',
            GPT2LMHeadModel: lambda layer: f'transformer.h.{layer}.ln_2'
        }
        return hookpoints[type(model)]
    
    @staticmethod
    def attention(model):
        hookpoints = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.attention',
            GPT2LMHeadModel: lambda layer: f'transformer.h.{layer}.attn'
        }
        return hookpoints[type(model)]
    
    @staticmethod
    def mlp(model):
        hookpoints = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.mlp',
            GPT2LMHeadModel: lambda layer: f'transformer.h.{layer}.mlp'
        }
        return hookpoints[type(model)]

    @staticmethod
    def from_address(address, model):

        def to_method(component):
            if component == 'layernorm_1':
                return Hookpoints.layernorm_1(model)
            if component == 'layernorm_2':
                return Hookpoints.layernorm_2(model)
            if component == 'attention':
                return Hookpoints.attention(model)
            if component == 'mlp':
                return Hookpoints.mlp(model)

        if ':' in address:
            address, _ = address.split(':')

        if 'layer' in address:
            _, layer, component = address.split('.')
            return to_method(component)(layer)
        else:
            return to_method(address)

        


class ModelConfig:

    @staticmethod
    def hidden_layers(model):
        _hidden_layers = {
            GPTNeoXForCausalLM: lambda: model.config.num_hidden_layers,
            GPT2LMHeadModel: lambda: model.config.n_layer
        }
        return _hidden_layers[type(model)]()
    
    @staticmethod
    def hidden_size(model):
        _hidden_size = {
            GPTNeoXForCausalLM: lambda: model.config.hidden_size,
            GPT2LMHeadModel: lambda: model.config.n_embd
        }
        return _hidden_size[type(model)]()
    

class Device:

    @staticmethod
    def device(model):
        return model.parameters().__next__().device
    
