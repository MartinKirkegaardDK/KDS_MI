import json
from matplotlib import pyplot as plt
from utils.plotting import plot_probe_results
from pathlib import Path

def run(
        probe_result_path_by_reg_lambda: dict[str,float],
        model_name: str
    ) -> None:
    """
    Plots the probes and saves them at: results/probe_confidence_intervals/*model_name*
    """
    output_name = list(probe_result_path_by_reg_lambda.values())[0].replace("results/data/probe_confidence_intervals/","")
    saved_path = "results/probe_confidence_intervals"
    # make figure
    num_reg_lambdas = len(probe_result_path_by_reg_lambda.keys())
    fig, axs = plt.subplots(ncols=2, nrows=num_reg_lambdas // 2 + int(num_reg_lambdas % 2 != 0), figsize=(20, 3*num_reg_lambdas))
    axs = axs.flatten()
    
    for idx, (reg_lambda, path) in enumerate(probe_result_path_by_reg_lambda.items()):

        # loads probe results
        with open(path, 'r') as file:
            probe_accuracies_by_layer = json.load(file)

        map_lab = probe_accuracies_by_layer.pop('map_label')

        plot_probe_results(
            accuracies_by_layer=probe_accuracies_by_layer,
            reg_lambda=reg_lambda,
            map_lab=map_lab,
            ax=axs[idx]
        )

    if "download" in model_name:
        model_name = model_name.split("/")[-1]

    fig.tight_layout()
    Path(saved_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{saved_path}/{output_name}.png")

