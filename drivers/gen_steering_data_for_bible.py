from collections import defaultdict
import pandas as pd
from utils.probe_confidence_intervals import model_setup
from utils.gen_steering_data_for_bible import gen_outputs, load_targeted_steering_vectors
from classes.datahandling import ParallelNSPDataset


def run(steering_vector_path: str, model_name: str, language1:str, language2:str, start_verse:int):
    """gradually computes the output to results/data/steering_data_bible

    Args:
        steering_vector_path (str): the steering vector you want to use
        model_name (str): the name of the llm
        language1 (str): the language you want to steer towards
        language2 (str): the initial language (currently only en as we dont have any other data)
        start_verse (int): the verse which the computations begins from
    """
    
    model, tokenizer, device = model_setup(model_name)
    bible_data = ParallelNSPDataset.from_tmx("data/bible-da-en.tmx",language1,language2)
    

    target, complement, combined = load_targeted_steering_vectors(steering_vector_path)

    
    #kør alle lag, kør hvert vers 5 gange og hav 
    layer = 15
    lambda_amount = 5
    bible_verse = 50

    for bible_verse in range(start_verse, 1000):
        temp_d = defaultdict(list)
        #we run each verse 5 times
        for _ in range(2):
            print(_)
            for lambda_amount in [2,5,10,15]:
                for layer in range(model.config.num_hidden_layers):
                        danish_predicted_output, english_predicted_output, danish_true_label,english_true_label = gen_outputs(bible_data, language1,language2,bible_verse,layer,combined,lambda_amount, model, tokenizer)
                        
                        temp_d["danish_predicted_output"].append(danish_predicted_output)
                        temp_d["english_predicted_output"].append(english_predicted_output)
                        temp_d["danish_true_label"].append(danish_true_label)
                        temp_d["english_true_label"].append(english_true_label)
                        temp_d["layer"].append(layer)
                        temp_d["lambda_amount"].append(lambda_amount)
                        temp_d["bible_verse"].append(bible_verse)

        df = pd.DataFrame(temp_d)
        df.to_csv(f"results/data/steering_data_bible/verse_{bible_verse}.csv", index = False)

if __name__ == "__main__":
    steering_vector_path = "steering_vectors/test_run_2/"
    model_name = "AI-Sweden-Models/gpt-sw3-356m"
    language1 = "da"
    langauge2 = "en"
    start_verse = 3
    run(steering_vector_path,model_name, language1, langauge2, start_verse)