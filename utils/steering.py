import torch
from transformers import PreTrainedTokenizerBase, GPTNeoXForCausalLM, GPT2Model

from classes.datahandling import TextClassificationDataset
from classes.hook_manager import HookManager

def generate_with_steering(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        layer: int,
        text_prompts: TextClassificationDataset | list[str],
        steering_vector: torch.Tensor,
        steering_lambda: int = 1,
        amount_samples: int = 10,
        cut_off: int = 10
    ) -> list[str]:
    '''
    Generates text from a set of prompts. The prompts will be cut up after a certain amount of tokens, and continued by the model under steering. If prompts are provided as TextClassificationDataset, they are automatically cut up randomly.

    Args:
        model: the torch model
        tokenizer: the model's tokenizer
        layer: the layer of model to attach steering vector
        text_prompts: a TextClassificationDataset containing the prompts OR a list of prompts
        steering_vector: the torch.Tensor steering vector
        steering_lambda: the scalar which will scale the steering vector before it being applied
        amount_sample: the amount of sentence generations to perform
        cut_off: at what token the prompts will be cut off

    Returns:
        a list of the output strings
    '''

    device = model.parameters().__next__().device

    #
    is_textclassdataset = isinstance(text_prompts, TextClassificationDataset)
    outputs = []

    with HookManager(model) as hook_manager:

        hook_manager.attach_residual_stream_activation_based_steering_vector(
            layer=layer,
            steering_vector=steering_vector,
            plus=True,
            scalar=steering_lambda,
            pre_mlp=False,
            pythia=True if isinstance(model, GPTNeoXForCausalLM) else False
        )
        
        for i in range(min(amount_samples, len(text_prompts))):
            if is_textclassdataset:
                text, _ = text_prompts[i]
            else:
                text = text_prompts[i]
    
            tokenized = tokenizer(text, return_tensors='pt').to(device)

            undecoded_output = model.generate(
                inputs=tokenized.input_ids[:, :cut_off] if is_textclassdataset else tokenized.input_ids, 
                max_length=100, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True
            )

            outputs.append(tokenizer.decode(undecoded_output[0]).replace('\n', '  '))

    return outputs


def loss_with_steering(
        model: GPTNeoXForCausalLM | GPT2Model,
        tokenizer: PreTrainedTokenizerBase,
        layer: int,
        prompt: str,
        continuation: str,
        steering_vector: torch.Tensor,
        steering_lambda: int = 1
    ) -> float:
    '''
    Computes the loss (next-token prediction) for a steered model, given a prompt and a true continuation. Returns only the loss on the continuation.

    Args:
        model: the torch model
        tokenizer: the model's tokenizer
        layer: the layer of model to attach steering vector
        prompt: the text prompt, given as context for prediction
        continuation: the ground truth continuation, used for computing loss
        steering_vector: the torch.Tensor steering vector
        steering_lambda: the scalar which will scale the steering vector before it being applied

    Returns:
        Average loss on continuation
    '''
    
    device = model.parameters().__next__().device

    with HookManager(model) as hook_manager:
        hook_manager.attach_residual_stream_activation_based_steering_vector(
            layer=layer,
            steering_vector=steering_vector,
            plus=True,
            scalar=steering_lambda,
            pre_mlp=False,
            pythia=True if isinstance(model, GPTNeoXForCausalLM) else False
        )

        # tokenizes the prompt and the continuation
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
        tokenized_continuation = tokenizer(continuation, return_tensors='pt').to(device)

        # combines them into one input
        tokenized_combined = torch.concat(
            (tokenized_prompt.input_ids, tokenized_continuation.input_ids), 
            dim=1
        )

        # computes output from model 
        output = model(tokenized_combined)

        # gets predictions and labels (need to be shifted by one)
        shift_logits = output.logits[:, :-1, :]
        shift_labels = tokenized_combined[:, 1:]

        # computes cross-entropy loss for all tokens individualle
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_by_token = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # isolates tokens from continuation and averages their loss
        loss_continuation = loss_by_token[-tokenized_continuation.input_ids.numel():]
        avg_loss_continuation = loss_continuation.mean().item()

        return avg_loss_continuation