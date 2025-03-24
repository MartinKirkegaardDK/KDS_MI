import pandas as pd
from matplotlib import pyplot as plt

def mean_score_by_layer(df: pd.DataFrame, lambda_amount: int):
    return df[df['lambda_amount'] == lambda_amount].groupby('layer')['english_prompt_score'].mean()

def run(file_path):

    # load data
    df = pd.read_csv(file_path)

    # find which lambdas
    lambdas = list(df['lambda_amount'].unique())
    lambdas.sort()

    # make fig
    fig, axs = plt.subplots(len(lambdas), 1,  figsize=(10, len(lambdas)*3))
    axs = axs.flatten()

    for ax, lambda_ in zip(axs, lambdas):
        mean_scores = mean_score_by_layer(df, lambda_)
        ax.bar(mean_scores.index, mean_scores)
        ax.set_ylim(0, 1)
        ax.set_title(f'score for lambda value {lambda_}')

    fig.tight_layout()
    