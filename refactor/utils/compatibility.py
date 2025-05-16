from transformers import GPTNeoXForCausalLM, GPT2LMHeadModel

class Hookpoints:
    
    @staticmethod
    def extraction(model, layer):
        _extraction = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.input_layernorm',
            GPT2LMHeadModel: lambda layer: f'transformer.h.{layer}.ln_1'
        }
        return _extraction[type(model)](layer)
    
    @staticmethod
    def steering(model, layer):
        _steering = {
            GPTNeoXForCausalLM: lambda layer: f'gpt_neox.layers.{layer}.mlp',
            GPT2LMHeadModel: lambda layer: f'gpt_neox.layers.{layer}.mlp'
        }
        return _steering[type(model)](layer)
    

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
    
