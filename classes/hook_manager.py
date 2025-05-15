import torch

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
    
    def attach_residstream_preln_hook(self, layer, pre_mlp=False, pythia=False):
        if pre_mlp:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.post_attention_layernorm'
            else:
                hookpoint = f'transformer.h.{layer}.ln_2'
        else:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.input_layernorm'
            else:
                hookpoint = f'transformer.h.{layer}.ln_1'
        
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
    
    def attach_residual_stream_activation_based_steering_vector(self, layer, steering_vector, plus, scalar = 1, pre_mlp=False, pythia=False):
        #This is martins' 
        
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
            if plus:
                activation = activation + steering_vector * scalar
            else:
                activation = activation - steering_vector * scalar
            return activation

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(steering_hook)
        )

    def attach_residual_stream_activation_based_steering_vector_preln(self, layer, steering_vector, plus, scalar = 1, pre_mlp=False, pythia=False):
        #This is martins' 
        
        if pre_mlp:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.post_attention_layernorm'
            else:
                hookpoint = f'transformer.h.{layer}.ln_2'
        else:
            if pythia:
                hookpoint = f'gpt_neox.layers.{layer}.input_layernorm'
            else:
                hookpoint = f'transformer.h.{layer}.ln_1'
        
        
        def steering_hook(module, input):
            activation = input[0]
            if plus:
                activation = activation + steering_vector * scalar
            else:
                activation = activation - steering_vector * scalar
            return activation

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(steering_hook)
        )

        
        

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

