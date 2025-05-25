import json
import pandas as pd
from utils.scandeval_results import get_confidence_interval, plot
from itertools import product
    

def main(scandeval_path: str,with_steering: bool,  model_names:list[str]|None):
    """
    creates the plots for the scandeval results.
    Saves the plots at results/scandeval/with(out)_steering_nlg.png
    model_name example: gpt_sw3_356m_with_steering_lambda_5
    """

    #scandeval = "results/scandeval/scandeval_benchmark_results_new.jsonl"


    # Read each line as a separate JSON object
    data = []
    with open(scandeval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # ignore empty lines
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    if model_names != None:
        df = df[df["model"].isin(model_names)]

    ds_filter = {
        "angry-tweets":"mcc",
        "dansk": "micro_f1",#Should we include the no misc ??
        "scala-da": "mcc",
        "scandiqa-da": "f1", #We use this as it is more lenient compared to exact matching
        "nordjylland-news": "bertscore", #bertscore apparently more aligns with human evals compared to rouge - scandeval
        "danske-talemaader": "mcc",
        "danish-citizen-tests": "mcc",
        "hellaswag-da": "mcc",
    }


    all_models = df.model.unique()
    all_datasets = ds_filter.keys()


    # Define what "no overlap" means — here: elements must not be equal
    combinations = [(a, b) for a, b in product(all_datasets, all_models) if a != b]

    li = []
    for ds, m in combinations:
        mask = (df.model == m) & (df.dataset == ds)
        try:
            metric = ds_filter[ds]
            task = df[mask].task.iloc[0]
            #value = df[mask].iloc[0].results["total"][metric]
            data = df[mask].iloc[0].results["raw"]["test"]

            min_value, average_value, max_value = get_confidence_interval(data, metric)
            li.append((m, ds, metric, min_value, average_value, max_value, task))
        except:
            print(ds, m)
        
    df = pd.DataFrame(li, columns=["Model", "Dataset", "Metric", "Min", "Avg","Max", "Task"])
    plot(df, with_steering)
    

if __name__ == "__main__":
    main("results/scandeval/scandeval_benchmark_results_new.jsonl", False)