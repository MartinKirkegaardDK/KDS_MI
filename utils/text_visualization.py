import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_text_viz(input_text:str,language_arg:str, model_name_temp:str,layer:int):
    
    # Parse the input data
    values = [float(t[0]) for t in input_text]
    languages = [t[1] for t in input_text]
    words = [t[2] for t in input_text]

    language_colors = {
        'en': 'royalblue',
        'is': 'crimson',
        'nb': 'limegreen',
        'sv': 'orange',
        "da": "pink"
    }
    
        # Use a distinct color palette
    colors = plt.cm.Set1(range(5))
    
    language_colors = {
        "da": colors[0],
        'en': colors[1],
        'is': colors[2],
        'sv': colors[3],
        'nb': colors[4]
    }
    
    

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    max_width = 12
    line_height = 0.6
    x, y = 0, 0

    def get_color_with_alpha(language, confidence):
        return language_colors.get(language, 'gray')

    for value, lang, word in zip(values, languages, words):
        word_width = len(word) * 0.15
        if x + word_width > max_width:
            x = 0
            y -= line_height

        rect = patches.Rectangle((x, y - 0.25), word_width, 0.5,
                                color=get_color_with_alpha(lang, value),
                                alpha=value,
                                edgecolor='black',
                                linewidth=0.5)
        ax.add_patch(rect)

        ax.text(x + word_width / 2, y, word,
                ha='center', va='center',
                color='black',
                fontweight='bold',
                fontsize=10)

        x += word_width + 0.1

    # Legend — positioned very close to text
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor='black', label=lang)
        for lang, color in language_colors.items()
    ]

    legend = ax.legend(handles=legend_elements,
                      loc='upper center',
                      bbox_to_anchor=(0.5, 0.05),  # Very close to text bottom
                      ncol=len(language_colors),
                      facecolor='white',
                      edgecolor='black',
                      framealpha=0.8,
                      labelcolor='black')

    # Title — positioned very close with no padding
    ax.set_title('Language Detection with Confidence Visualization',
                color='black', fontsize=14, pad=0)  # No padding

    # Adjust plot limits - decrease text-title gap, slightly more bottom space
    ax.set_xlim(-0.5, max_width + 0.5)
    ax.set_ylim(y - 0.5, 0.4)  # Less space at top (closer title), slightly more at bottom

    # Adjust margins: bring title much closer, give legend slightly more space
    plt.subplots_adjust(top=0.98, bottom=0.12, left=0.05, right=0.95)
    
    plt.savefig(f'results/text_visualization/{model_name_temp}_{language_arg}_{str(layer)}.png',
                facecolor='white', dpi=300, bbox_inches='tight')
    plt.show()