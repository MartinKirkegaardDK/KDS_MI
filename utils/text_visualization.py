import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_text_viz(input_text:str,language_arg:str, model_name_temp:str,layer:int):
    
    # Parse the input data

    values = [float(t[0]) for t in input_text]
    languages = [t[1] for t in input_text]
    words = [t[2] for t in input_text]

    # Define colors for each language
    language_colors = {
        'en': 'royalblue',
        'is': 'crimson',
        'nb': 'limegreen',
        'sv': 'orange',
        "da": "pink"
    }

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')  # Dark background
    ax.set_facecolor('#1a1a1a')

    # Turn off axes
    ax.axis('off')

    # Define the maximum width and height for the visualization
    max_width = 12
    line_height = 0.6
    x, y = 0, 0

    # Define a function to get color with alpha based on confidence
    def get_color_with_alpha(language, confidence):
        base_color = language_colors.get(language, 'gray')
        # Use the confidence as alpha for the color intensity
        return base_color

    # Display text with colored backgrounds
    for i, (value, lang, word) in enumerate(zip(values, languages, words)):
        # Calculate the width of this word (including space)
        word_width = len(word) * 0.15
        
        # Check if we need to start a new line
        if x + word_width > max_width:
            x = 0
            y -= line_height
        
        # Create a rectangle with color based on language and intensity based on value
        rect = patches.Rectangle((x, y-0.25), word_width, 0.5, 
                                color=get_color_with_alpha(lang, value),
                                alpha=value,  # Use the tensor value as opacity
                                edgecolor='white',
                                linewidth=0.5)
        ax.add_patch(rect)
        
        # Add the word
        ax.text(x + word_width/2, y, word, 
                ha='center', va='center', 
                color='white',
                fontweight='bold',
                fontsize=10)
        
        # Move to the next position
        x += word_width + 0.1

    # Create a legend
    legend_elements = []
    for lang, color in language_colors.items():
        legend_elements.append(patches.Patch(facecolor=color, edgecolor='white', 
                                            label=f'{lang}'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
            ncol=len(language_colors), facecolor='#1a1a1a', edgecolor='white', framealpha=0.8,
            labelcolor='white')

    # Set title
    ax.set_title('Language Detection with Confidence Visualization', color='white', fontsize=14)

    # Set limits to show all content
    ax.set_xlim(-0.5, max_width + 0.5)
    ax.set_ylim(y-1, 1)

    plt.tight_layout()
    plt.savefig(f'results/text_visualization/{model_name_temp}_{language_arg}_{str(layer)}.png', facecolor='#1a1a1a', dpi=300)
    plt.show()
