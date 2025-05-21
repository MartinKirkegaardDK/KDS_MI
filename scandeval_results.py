from numpy import average
import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_confidence_interval(data: list,metric: str):
    li = [x[metric] for x in data]
    return min(li), average(li), max(li)


def plot(df:pd.DataFrame, with_steering:bool):
    if with_steering:
        var = "with"
    else:
        var = "without"
    # Combine Dataset, Metric, and Task for clarity
    df["Dataset_Metric"] = df["Dataset"] + " | " + df["Metric"] + " | " + df["Task"]

    # Define task groups
    nlu_tasks = [
        "sentiment-classification",
        "named-entity-recognition",
        "linguistic-acceptability",
        "reading-comprehension"
    ]
    nlg_tasks = [
        "summarization",
        "knowledge",
        "common-sense Reasoning"
    ]

    # Filter data
    df_nlu = df[df["Task"].isin(nlu_tasks)]
    df_nlg = df[df["Task"].isin(nlg_tasks)]

    # Get unique models and assign unique colors
    unique_models = df["Model"].unique()
    cmap = plt.get_cmap('tab10')  # Or 'tab20' if more than 10 models
    model_colors = {model: cmap(i % cmap.N) for i, model in enumerate(unique_models)}

    # Get unique Dataset_Metric values
    nlu_metrics = df_nlu["Dataset_Metric"].unique()
    nlg_metrics = df_nlg["Dataset_Metric"].unique()

    # ---- FIGURE 1: NLU TASKS ----
    fig_nlu, axes_nlu = plt.subplots(1, len(nlu_metrics), figsize=(20, 6))
    if len(nlu_metrics) == 1:
        axes_nlu = [axes_nlu]
    for i, metric in enumerate(nlu_metrics):
        ax = axes_nlu[i]
        subset = df_nlu[df_nlu["Dataset_Metric"] == metric]

        for j, (_, row) in enumerate(subset.iterrows()):
            avg = row["Avg"]
            min_val = row["Min"]
            max_val = row["Max"]
            color = model_colors[row["Model"]]

            # Plot confidence interval line with dot at avg
            ax.errorbar(
                x=j, y=avg,
                yerr=[[avg - min_val], [max_val - avg]],
                fmt='o',  # Only the marker
                color=color,
                capsize=5,
                markersize=6,
                elinewidth=2
            )

        ax.set_title(f"NLU: {metric}", fontsize=12)
        ax.set_ylabel("Score")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(subset)))
        #ax.set_xticklabels(subset["Model"], rotation=90)


    legend_patches = [mpatches.Patch(color=color, label=model) for model, color in model_colors.items()]
    fig_nlu.legend(handles=legend_patches, loc='upper center', ncol=len(unique_models),
                bbox_to_anchor=(0.5, 1.05))
    plt.suptitle("Natural Language Understanding (NLU) Tasks", fontsize=16, y=1.1)
    plt.tight_layout()
    plt.savefig(f"results/scandeval/{var}_steering_nlu.png",bbox_inches='tight')
    #plt.show()

    # ---- FIGURE 2: NLG TASKS ----
    fig_nlg, axes_nlg = plt.subplots(1, len(nlg_metrics), figsize=(20, 6))
    if len(nlg_metrics) == 1:
        axes_nlg = [axes_nlg]

    for i, metric in enumerate(nlg_metrics):
        ax = axes_nlg[i]
        subset = df_nlg[df_nlg["Dataset_Metric"] == metric]

        for j, (_, row) in enumerate(subset.iterrows()):
            avg = row["Avg"]
            min_val = row["Min"]
            max_val = row["Max"]
            color = model_colors[row["Model"]]

            # Plot confidence interval line with dot at avg
            ax.errorbar(
                x=j, y=avg,
                yerr=[[avg - min_val], [max_val - avg]],
                fmt='o',  # Only the marker
                color=color,
                capsize=5,
                markersize=6,
                elinewidth=2
            )

        ax.set_title(f"NLG: {metric}", fontsize=12)
        ax.set_ylabel("Score")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(subset)))
        #ax.set_xticklabels(subset["Model"], rotation=90)

    fig_nlg.legend(handles=legend_patches, loc='upper center', ncol=len(unique_models),
                bbox_to_anchor=(0.5, 1.05))
    plt.suptitle("Natural Language Generation (NLG) Tasks", fontsize=16, y=1.1)
    plt.tight_layout()
    plt.savefig(f"results/scandeval/{var}_steering_nlg.png",bbox_inches='tight')
    #plt.show()
    
    

def main():
    """
    creates the plots for the scandeval results.
    Saves the plots at results/scandeval/with(out)_steering_nlg.png
    """

    scandeval = "results/scandeval/scandeval_benchmark_results_new.jsonl"


    # Read each line as a separate JSON object
    data = []
    with open(scandeval, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # ignore empty lines
                data.append(json.loads(line))

    df = pd.DataFrame(data)

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
        metric = ds_filter[ds]
        task = df[mask].task.iloc[0]
        #value = df[mask].iloc[0].results["total"][metric]
        data = df[mask].iloc[0].results["raw"]["test"]

        min_value, average_value, max_value = get_confidence_interval(data, metric)
        li.append((m, ds, metric, min_value, average_value, max_value, task))
        
    df = pd.DataFrame(li, columns=["Model", "Dataset", "Metric", "Min", "Avg","Max", "Task"])
    plot(df, False)
    

if __name__ == "__main__":
    main()