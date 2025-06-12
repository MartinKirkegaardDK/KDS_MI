from collections import defaultdict
from src.utils.old.probe_confidence_intervals import model_setup
from src.utils.old.create_bible_data import load_targeted_steering_vectors, gen_outputs
from classes.datahandling import ParallelNSPDataset
import pandas as pd

def run(model_name: str, 
        language1: str, 
        language2: str, 
        steering_vector_path: str, 
        lambda_search_space: list, 
        start_verse:int, 
        end_verse: int):
    """
    danish_predicted_output is the output of model with a danish verse as prompt
    english_predicted_output is the output of the model with a steering vector danish with an english verse as input.
    english_predicted_output should hopefully be in danish if a danish steering vector is inserted
    
    Args:
        model_name (str): name of the model example: "AI-Sweden-Models/gpt-sw3-356m"
        language1 (str): example: "da"
        language2 (str): exmaple: "en"
        steering_vector_path (str): the path to an extracted steering vector. Example: "steering_vectors/test_run_2/"
        lambda_search_space (list): the lambdas you want to try. Example: [2,5,10,15]
        start_verse (int): the verse you want to begin from
        end_verse (int): the verse you want to end at.
        example: if start verse 20 and end verse 80, then we compute the bible data for versen 20 to 80
    """
    
    
    
    model, tokenizer, device = model_setup(model_name)
    
    bible_data = ParallelNSPDataset.from_tmx("data/bible-da-en.tmx",language1,"en")

    target, complement, combined = load_targeted_steering_vectors(steering_vector_path,device)

    for bible_verse in range(start_verse, end_verse):
        temp_d = defaultdict(list)
        #we run each verse 2 times
        for _ in range(2):
            print(_)
            for lambda_amount in lambda_search_space:
                
                
                #for layer in range(model.config.num_hidden_layers):
                layer = 15
                danish_predicted_output, english_predicted_output, danish_true_label,english_true_label = gen_outputs(bible_data, language1,language2,bible_verse,layer,combined,lambda_amount, model,tokenizer,device)
                
                temp_d["danish_predicted_output"].append(danish_predicted_output)
                temp_d["english_predicted_output"].append(english_predicted_output)
                temp_d["danish_true_label"].append(danish_true_label)
                temp_d["english_true_label"].append(english_true_label)
                temp_d["layer"].append(layer)
                temp_d["lambda_amount"].append(lambda_amount)
                temp_d["bible_verse"].append(bible_verse)
        df = pd.DataFrame(temp_d)
        df.to_csv(f"results/data/steering_data_bible/verse_{bible_verse}_no_lambda.csv", index = False)





if __name__ == "__main__":
    steering_vector_path = "steering_vectors/test_run_2/"
    model_name = "AI-Sweden-Models/gpt-sw3-356m"
    language1 = "da"
    langauge2 = "en"
    start_verse = 3
    end_verse = 1000
    lambda_search_space = [2,5,10,15]
    
    run(model_name,language1, steering_vector_path, lambda_search_space, start_verse,end_verse)
    
    
    
    
    