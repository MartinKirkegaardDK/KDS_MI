import json
from matplotlib import pyplot as plt
from utils.plotting import plot_probe_results

def run(
        probe_result_path_by_reg_lambda: dict[float, str]
    ) -> None:

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

    fig.tight_layout()

