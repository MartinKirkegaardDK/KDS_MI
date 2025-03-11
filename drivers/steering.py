import torch
from pathlib import Path

from utils.probe_confidence_intervals import model_setup
from utils.preprocessing import load_txt_data
from utils.steering import generate_with_steering

def run(
        steering_vector_path,
        steering_lambda,
        affected_language, 
        layer,
        model_name,
        data_folder
):

    # loads model
    model, tokenizer, device = model_setup(model_name)

    # loads data
    data_folder = Path(data_folder)
    ds = load_txt_data(
        file_paths={
            'da': data_folder / 'da.txt',
            'en': data_folder / 'en.txt',
            'sv': data_folder / 'sv.txt',
            'nb': data_folder / 'nb.txt',
            'is': data_folder / 'is.txt'
        },
        file_extension='txt'
    )

    # loads steering vector
    steering_vector_path = Path(steering_vector_path)
    steering_vector = torch.load(str(steering_vector_path))

    
    # gets prompts that will be affected by steering
    text_prompts = ds.filter_by_language(affected_language)

    # generates n completed, steered prompts
    amount_samples = 10

    outputs = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        text_prompts=text_prompts,
        steering_vector=steering_vector,
        steering_lambda=steering_lambda,
        amount_samples=amount_samples,
        cut_off=10
    )

    for output in outputs:
        print(output, '\n\n')





