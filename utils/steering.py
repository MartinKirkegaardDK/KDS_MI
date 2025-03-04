import torch
from classes.datahandling import TextClassificationDataset
from classes.hook_manager import HookManager

def generate_with_steering(
        model,
        tokenizer,
        layer,
        text_prompts: TextClassificationDataset,
        steering_vector: torch.Tensor,
        steering_lambda: int = 1,
        amount_samples: int = 10,
        cut_off: int = 10) -> list[str]:
    '''
    Generates text from a set of prompts. The prompts will be cut up after a certain amount of tokens, and continued by the model under steering

    Args:
        model: the torch model
        tokenizer: its tokenizer
        layer: the layer of model to attach steering vector
        text_prompts: a TextClassificationDataset containing the prompts
        steering_vector: the torch.Tensor steering vector
        steering_lambda: the scalar which will scale the steering vector before it being applied
        amount_sample: the amount of sentence generations to perform
        cut_off: at what token the prompts will be cut off

    Returns:
        a list of the output strings
    '''

    device = model.parameters().__next__().device
    outputs = []

    with HookManager(model) as hook_manager:
        hook_manager.attach_residual_stream_activation_based_steering_vector(
            layer=layer,
            steering_vector=steering_vector,
            plus=True,
            scalar=steering_lambda,
            pre_mlp=False,
            pythia=False
        )

        for i in range(amount_samples):
            text, _ = text_prompts[i]
            tokenized = tokenizer(text, return_tensors='pt').to(device)
            undecoded_output = model.generate(
                tokenized.input_ids[:, :cut_off], 
                max_length=100, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True
            )
            outputs.append(tokenizer.decode(undecoded_output[0]).replace('\n', '  '))

    return outputs
