import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs, getcwd
import pandas as pd

sns.set_style("whitegrid")


def plot_train_val(y_train, y_val, epochs, params, step=1):
    x = list(range(1, epochs + 1))
    fig, ax = plt.subplots()
    ax.set_xticks(list(range(0, epochs + 1, step)))
    plt.plot(x, y_train, color='b', label='train')
    plt.plot(x, y_val, color='r', label='validation')
    plt.title(params["title"])
    plt.ylabel(params["ylabel"])
    plt.xlabel(params["xlabel"])
    plt.legend(loc='upper right')
    plt.pause(1)
    makedirs(params["folder_path"], exist_ok=True)
    full_path = path.join(params["folder_path"], params["file_name"])
    plt.savefig(full_path)
    plt.close()


def make_plots(logged_values, model_name='ez'):
    # params = dict()
    # params["xlabel"] = "epochs"
    # params["folder_path"] = path.join(getcwd(), "Plots", model_name)
    # makedirs(params["folder_path"], exist_ok=True)
    folder_path = path.join(getcwd(), "Plots")
    makedirs(path.join(getcwd(), "Plots", model_name), exist_ok=True)
    file_path = path.join(folder_path, f'{model_name}.png')
    fig, ax = plt.subplots(4, sharex=True)
    ax = ax.flat

    # causal = pd.DataFrame(logged_values['causal'])
    # anti_causal = pd.DataFrame(logged_values['anticausal'])
    # total = pd.DataFrame(logged_values['total'])
    # symmetry = pd.DataFrame(logged_values['symmetry'])
    # sns.lineplot(data=causal, ax=ax[0])
    # ax[0].set_title('causal')
    # sns.lineplot(data=anti_causal, ax=ax[1])
    # ax[1].set_title('anti causal')
    # sns.lineplot(data=total, ax=ax[2])
    # ax[2].set_title('total')
    # sns.lineplot(data=symmetry, ax=ax[3])
    # ax[3].set_title('symmetry')
    # for p in ax:
    #     p.set_ylim(-0.05, 1.05)
    # plt.pause(1)
    plot_titles = [plot_title for plot_title in logged_values.keys()]
    dict_hierarchy = {(plot_title, plot_type): values for plot_title, train_val_dict in logged_values.items() for
                     plot_type, values in train_val_dict.items()}
    df_plot = pd.DataFrame(dict_hierarchy)
    df_plot.set_index(pd.Index(range(1, df_plot.shape[0]+1)), inplace=True)
    for i, plot_title in enumerate(plot_titles):
        sns.lineplot(data=df_plot[plot_title], ax=ax[i])
        ax[i].set_title(plot_title)
        ax[i].set_ylim(-0.05, 1.05)
    ax[-1].set_xlabel('Epoch')
    plt.savefig(file_path)
    plt.close()
    # plt.show()
    # plt.pause(1)

    a = 0
    # for causal_type in error_dict.keys():
    #     params["ylabel"] = "validation error"
    #     params["file_name"] = f'error_{causal_type}.png'
    #     params["title"] = f"{causal_type} validation error"
    #
    #     y_train = error_dict[causal_type]['train']
    #     y_val = error_dict[causal_type]['validation']
    #     plot_train_val(y_train, y_val, epochs, params, step=step)

    # params["ylabel"] = "0.5 (1-NCC(x,y) + NCC(y,x))"
    # params["file_name"] = 'symmetry_check.png'
    # params["title"] = f"symmetry check"
    # y_train = symmetry_check_dict['train']
    # y_val = symmetry_check_dict['validation']
    # plot_train_val(y_train, y_val, epochs, params, step=step)
