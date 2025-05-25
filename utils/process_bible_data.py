import os
import pandas as pd
from fasttext.FastText import _FastText


def combine_bible_data(bible_dat_path: str, language_prediction_model: _FastText, language_label:str):
    """_summary_

    Args:
        bible_dat_path (str): the path where the bible data is saved
        language_prediction_model (_FastText): The language model we use to predict which language text is
        language_label (str): example: __label__dan
    """
    
    li = []
    
    path = bible_dat_path
    
    for file in os.listdir(path):
        if ("combined" in file) or (".DS_Store" in file):
            continue
        #print(file)
        li.append(pd.read_csv(f"{path}/{file}"))
    loaded_df = pd.concat(li)
    danish_prompt_score_list = []
    english_prompt_score_list =[]
    danish_language_prediction_on_english_steered = []
    
    for _, row in loaded_df.iterrows():
        danish_prompt = row["danish_predicted_output"]
        danish_prompt = danish_prompt.replace("\n","")
        prediction = predict_language(language_prediction_model,language_label,danish_prompt)
        danish_prompt_score_list.append(round(prediction, 2))
        
        english_prompt = row["english_predicted_output"]
        prediction = predict_language(language_prediction_model,"__label__eng",english_prompt)
        english_prompt_score_list.append(round(prediction,2))
        
        english_prompt = row["english_predicted_output"]
        prediction = predict_language(language_prediction_model,"__label__dan",english_prompt)
        danish_language_prediction_on_english_steered.append(round(prediction,2))
        
    #danish_prompt_score is the probability of the language being danish.
    #This is computed on the model getting danish as input. The idea is we want a baseline for the amount of time
    #the model predicts danish
    
    #english_prompt_score is the model steered towards danish but having english as input. Here we can see
    #how often the model still replies in english
    
    #danish_language_prediction_on_english_steered is to test how often the model being steered towards danish
    #with english input is actually danish
    loaded_df["danish_prompt_score"] = danish_prompt_score_list
    loaded_df["english_prompt_score"] = english_prompt_score_list
    loaded_df["danish_language_prediction_on_english_steered"] = danish_language_prediction_on_english_steered
    loaded_df.to_csv(path + "bible_data_combined.csv", index = False)
    

def predict_language(language_prediction_model:_FastText, language:str, text:str):
    """_summary_

    Args:
        language_prediction_model (_type_): the language model
        language (str): example of getting the score of english "__eng"
        text (str): the text you want to predict on
    """
    supported_languages = ('__label__dan',
    '__label__nob',
    '__label__sma',
    '__label__eng',
    '__label__nno',
    '__label__isl',
    '__label__sms',
    '__label__smj',
    '__label__swe',
    '__label__sme',
    '__label__smn',
    '__label__fin',
    '__label__fao')
    if language not in supported_languages:
        raise  ValueError(f"language {language} is not in supported language format: {supported_languages}")
    labels, score = language_prediction_model.predict(text, k = 13)
    for i in range(len(labels)):
        #print(language, labels[i], score[i])
        if language in labels[i]:
            return score[i]
    