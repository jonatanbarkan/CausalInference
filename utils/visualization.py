import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_style("whitegrid")


def make_plots(logged_values, plot_path: str, model_name: str):
    file_path = os.path.join(plot_path, f'{model_name}_accuracy.png')
    fig, ax = plt.subplots(4, sharex=True, dpi=300)
    ax = ax.flat

    plot_titles = list(logged_values.keys())
    dict_hierarchy = {(plot_title, plot_type): values for plot_title, train_val_dict in logged_values.items() for
                      plot_type, values in train_val_dict.items()}
    df_plot = pd.DataFrame(dict_hierarchy)
    df_plot.set_index(pd.Index(range(1, df_plot.shape[0] + 1)), inplace=True)
    for i, plot_title in enumerate(plot_titles):
        sns.lineplot(data=df_plot[plot_title], ax=ax[i])
        # ax[i].set_title(plot_title)
        ax[i].set_title(plot_title)
        ax[i].set_ylabel('' if plot_title.lower() in ['symmetry'] else r'1-acc')
        ax[i].set_ylim(-0.05, 1.05)
    ax[-1].set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    df_plot.to_csv(os.path.join(plot_path, f'{model_name}_accuracy.csv'))

