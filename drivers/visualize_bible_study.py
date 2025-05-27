import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import textwrap
import pandas as pd

def run(file_path,model_name_temp):

   # df = pd.read_csv("results/data/steering_data_bible/bible_data_combined.csv")
    df = pd.read_csv(file_path)
    df = df[(df.layer == 15) & (df.lambda_amount == 5)]

    # Columns for each KDE plot
    kde1_cols = [
        "danish_language_prediction_on_english_steered",
        "danish_score_on_english_prompt_without_steering"
    ]
    kde2_cols = [
        "english_prompt_score",
        "english_predicted_output_without_steering_score"
    ]

    # Titles
    titles = {
        "danish_language_prediction_on_english_steered": "English prompt, danish output, with steering",
        "danish_score_on_english_prompt_without_steering": "English prompt, danish output, no steering",

        "english_prompt_score": "English prompt, english output, with steering",
        "english_predicted_output_without_steering_score": "English prompt, english output, no steering"
    }

    # Color scheme using matplotlib's Set1 colormap
    #colors = plt.cm.Set1(range(5))
    color_map = {
        "danish_language_prediction_on_english_steered": "red",
        "danish_score_on_english_prompt_without_steering": "grey",
        "english_prompt_score": "red",
        "english_predicted_output_without_steering_score": "grey",
    }

    #Farve 1 english_predicted_output_without_steering_score

    # Create 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Function to plot KDE
    def plot_kde(ax, cols):
        for col in cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue  # Skip empty columns
            kde = gaussian_kde(data)
            x_vals = np.linspace(min(data), max(data), 200)
            y_vals = kde(x_vals)
            ax.plot(x_vals, y_vals, label="\n".join(textwrap.wrap(titles[col], width=40)), color=color_map[col])
            ax.fill_between(x_vals, y_vals, alpha=0.3, color=color_map[col])

    # Plot 1
    plot_kde(axes[0], kde1_cols)
    axes[0].set_title("KDE: English prompt with danish output")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Plot 2
    plot_kde(axes[1], kde2_cols)
    axes[1].set_title("KDE: English prompt with english output")
    axes[1].set_xlabel("Value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"results/bible_study/{model_name_temp}.png")
    #plt.show()
