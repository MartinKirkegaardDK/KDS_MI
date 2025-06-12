import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.new.hooking import HookManager, HookAddress
from src.utils.new.compatibility import *
from src.utils.new.data import load_antibiotic_data

from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict




def plot_logit_lens():
    model_url = 'EleutherAI/pythia-14m'
    device = 'cpu'

    model_url = 'AI-Sweden-Models/gpt-sw3-356m'
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(model_url).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_url)


    extracted = dict()

    with HookManager(model) as hook_manager:
        for hook_address in HookAddress:
            for layer in range(ModelConfig.hidden_layers(model)):
                extracted[hook_address.layer(layer)] = hook_manager.extract(hook_address.layer(layer))


        input_ = 'I Begyndelsen skabte Gud Himmelen og Jorden. Og Jorden var øde og tom, og der var Mørke over Verdensdybet.'
        tokenized = tokenizer(input_, return_tensors='pt').to(device)

        model.forward(tokenized.input_ids)


    avg_norm_logits = []
    token_preds = []
    norm_preds = []

    hook_address = HookAddress.layernorm_1_pre
    for layer in range(ModelConfig.hidden_layers(model)):
        logits_ln = model.get_submodule('transformer.ln_f').forward(extracted[hook_address.layer(layer)][0])
        logits = model.get_submodule('lm_head').forward(logits_ln)
        norm_preds_ = [torch.norm(logit).item() for logit in logits]
        norm_preds.append(norm_preds_)
        token_preds_ = [tokenizer.decode(torch.argmax(logit)) for logit in logits]
        token_preds.append(token_preds_)
        avg_norm_logits.append(mean(norm_preds_))



    # convert to numpy arrays for easier manipulation
    tokens = np.array([token_pred[-10:] for token_pred in token_preds])
    norms = np.array([norm_pred[-10:] for norm_pred in norm_preds])

    # create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # log transform the norms for better visualization
    log_norms = np.log10(norms)

    # create heatmap
    im = ax.imshow(log_norms, cmap='viridis', aspect='auto', interpolation='nearest')

    # set the ticks and labels
    ax.set_xticks(range(len(tokens[0])))
    ax.set_yticks(range(len(tokens)))

    # add token labels
    for i in range(len(tokens)):
        for j in range(len(tokens[0])):
            # handle empty tokens
            token_text = tokens[i, j] if tokens[i, j] else '∅'
            # adjust text color based on background intensity
            text_color = 'white'# if log_norms[i, j] < np.median(log_norms) else 'black'
            ax.text(j, i, token_text, ha='center', va='center', 
                fontsize=8, color=text_color, weight='bold')

    # labels and title
    ax.set_xlabel('token position', fontsize=12)
    ax.set_ylabel('layer', fontsize=12)
    ax.set_title('logit lens: predicted tokens by layer', fontsize=14)

    # add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(logit norm)', rotation=270, labelpad=20)

    # invert y axis so input is at bottom
    ax.invert_yaxis()

    # adjust layout
    plt.tight_layout()
    plt.savefig('results/figures/logit_lens/logit_lens.png', bbox_inches='tight', dpi=300)



def plot_norm_of_logits():

    model_url = 'EleutherAI/pythia-14m'
    device = 'cpu'

    model_url = 'AI-Sweden-Models/gpt-sw3-356m'
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(model_url).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    data = load_antibiotic_data()

    hook_address = HookAddress.layernorm_1_pre
    avg_norm_logits = defaultdict(list)

    max_its = 200

    for i, (input_, _) in enumerate(data):

        if i >= max_its:
            break
        extracted = dict()
        with HookManager(model) as hook_manager:
            for layer in range(ModelConfig.hidden_layers(model)):
                extracted[hook_address.layer(layer)] = hook_manager.extract(hook_address.layer(layer))

            tokenized = tokenizer(input_, return_tensors='pt').to(device)
            model.forward(tokenized.input_ids)

        for layer in range(ModelConfig.hidden_layers(model)):
            logits_ln = model.get_submodule('transformer.ln_f').forward(extracted[hook_address.layer(layer)][0])
            logits = model.get_submodule('lm_head').forward(logits_ln)
            norm_preds_ = [torch.norm(logit, 2).item() for logit in logits]
            avg_norm_logits[layer].append(mean(norm_preds_))


    fig, ax = plt.subplots(1, 1, figsize=(7, 3))


    layers = list(range(ModelConfig.hidden_layers(model)))
    avgs = [mean(avg_norm_logits[layer]) for layer in layers]
    ax.plot(layers, avgs)
    ax.set_ylim(750,2200)
    ax.set_ylabel('norm of logits')
    ax.set_xlabel('layer')
    ax.set_title('L2 norm of logit lens outputs')

    fig.tight_layout()
    fig.savefig('results/figures/logit_lens/logit_lens_norm.png', bbox_inches='tight', dpi=300)