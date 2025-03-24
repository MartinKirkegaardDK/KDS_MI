from utils.process_bible_data import combine_bible_data
from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download
import fasttext


def run(bible_dat_path: str, language_label: str):
    
    language_prediction_model = fasttext.load_model(hf_hub_download("NbAiLab/nb-nordic-lid", "nb-nordic-lid.ftz"))
    combine_bible_data(bible_dat_path, language_prediction_model, language_label)
    
    

if __name__ == "__main__":
    path = "results/data/steering_data_bible/"
    language_label = "__dan__"
    run(path, language_label)