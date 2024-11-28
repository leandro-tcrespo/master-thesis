import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_lime(exp, out_file):
    top_label = exp.top_labels[0]

    # Get predictions from explanation object
    pred_probas = exp.predict_proba
    class_names = exp.class_names

    # Create the plot using LIME's built-in visualization
    fig = exp.as_pyplot_figure(label=top_label)

    # Get the current axis
    ax = plt.gca()

    # Create two legends side by side at the bottom
    color_legend = ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Supports prediction'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Contradicts prediction')
        ],
        bbox_to_anchor=(0.3, -0.15),
        loc='upper center',
        ncol=1
    )

    ax.add_artist(color_legend)

    prob_legend = ax.legend(
        handles=[Line2D([], [], color='none', label=f'{name}: {prob:.2f}',
                        markerfacecolor='none', markeredgecolor='none')
                 for name, prob in zip(class_names, pred_probas)],
        title='Prediction Probabilities',
        bbox_to_anchor=(0.7, -0.15),
        loc='upper center',
        ncol=2,
        handletextpad=-2
    )

    plt.tight_layout()
    plt.savefig(out_file)
    return fig
