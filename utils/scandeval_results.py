from numpy import average
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

def get_confidence_interval(data: list,metric: str):
    li = [x[metric] for x in data]
    return min(li), average(li), max(li)



def plot_cool(df:pd.DataFrame):


    # --- Setup base model parsing functions ---
    def get_base_model_name(model_name):
        return model_name.split('_with_')[0].split('-lambda')[0]

    def is_steering_variant(model_name):
        return '_with_steering_lambda' in model_name or '-lambda' in model_name

    df['BaseModel'] = df['Model'].apply(get_base_model_name)

    # --- Task grouping ---
    nlu_tasks = [
        "sentiment-classification",
        "named-entity-recognition",
        "linguistic-acceptability",
        "reading-comprehension"
    ]
    nlg_tasks = [
        "summarization",
        "knowledge",
        "common-sense-reasoning"
    ]

    # Ensure consistent formatting
    df['Task'] = df['Task'].str.lower().str.strip()
    df['Dataset'] = df['Dataset'].str.strip()

    # Group by (Task, Dataset)
    df['Task-Dataset'] = df['Task'] + " | " + df['Dataset']
    task_dataset_groups = df.groupby(['Task', 'Dataset'])

    # Separate into NLG and NLU plots
    nlg_plots = [(task, dataset) for (task, dataset) in task_dataset_groups.groups.keys() if task in nlg_tasks]
    nlu_plots = [(task, dataset) for (task, dataset) in task_dataset_groups.groups.keys() if task in nlu_tasks]

    # Combine for plotting layout
    total_plots = max(len(nlg_plots), len(nlu_plots))
    fig, axs = plt.subplots(total_plots, 2, figsize=(13, 2 * total_plots))

    # Adjust layout to make room for legend at bottom
    plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)

    # --- Define 4 base colors ---
    manual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    unique_base_models = sorted(df['BaseModel'].unique())
    base_model_color_map = {
        base_model: manual_colors[i % len(manual_colors)]
        for i, base_model in enumerate(unique_base_models)
    }

    def get_model_color(model_name):
        base_model = get_base_model_name(model_name)
        base_rgb = np.array(mcolors.to_rgb(base_model_color_map[base_model]))
        return base_rgb * 0.6 if is_steering_variant(model_name) else base_rgb

    # Store legend elements globally
    global legend_handles, legend_labels

    legend_handles = []
    legend_labels = []

    # --- Helper to plot a single (task, dataset) on given axis ---
    def plot_task_dataset(ax, task, dataset):
        
        
        subset = df[(df['Task'] == task) & (df['Dataset'] == dataset)]
        grouped = subset.groupby('BaseModel')

        current_x = 0
        spacing = 1.5

        for base_model, group in grouped:
            group_sorted = group.sort_values('Model')
            for _, row in group_sorted.iterrows():
                yerr = [[row['Avg'] - row['Min']], [row['Max'] - row['Avg']]]
                
                handle = ax.errorbar(
                    x=[current_x], y=[row['Avg']],
                    yerr=yerr, fmt='o', capsize=5,
                    color=get_model_color(row['Model'])
                )
                
                # Collect unique legend elements
                if base_model not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(base_model)
                
                current_x += 1
            current_x += spacing

        title = f"{task} | {dataset}"
        ax.set_title(title)
        ax.set_ylabel(subset['Metric'].iloc[0])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.spines['bottom'].set_visible(True)

    # --- Final plot rendering ---
    for row in range(total_plots):
        # Left: NLG task
        if row < len(nlg_plots):
            task, dataset = nlg_plots[row]
            plot_task_dataset(axs[row, 0], task, dataset)
        else:
            axs[row, 0].axis('off')  # hide unused subplot

        # Right: NLU task
        if row < len(nlu_plots):
            task, dataset = nlu_plots[row]
            plot_task_dataset(axs[row, 1], task, dataset)
        else:
            axs[row, 1].axis('off')  # hide unused subplot

    # Add single legend at bottom center
    fig.legend(legend_handles, legend_labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.05),
            ncol=2,  # horizontal layout
            frameon=True)

    #plt.tight_layout()
    plt.savefig(f"results/scandeval/scandeval_plots.png",bbox_inches='tight')

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
    fig_nlu.legend(handles=legend_patches, loc='upper right', ncol=1,
                bbox_to_anchor=(0, 0.4))
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
    