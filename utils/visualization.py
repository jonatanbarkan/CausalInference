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


def make_plots(error_dict, symmetry_check_dict, epochs, model_type='NCC', step=1):
    params = dict()
    params["xlabel"] = "epochs"
    params["folder_path"] = path.join(getcwd(), "Plots", model_type)
    fig, ax = plt.subplots(4)
    ax = ax.flat
    causal = pd.DataFrame(error_dict['causal'])
    anti_causal = pd.DataFrame(error_dict['anticausal'])
    total = pd.DataFrame(error_dict['total'])
    symmetry = pd.DataFrame(symmetry_check_dict)
    sns.lineplot(data=causal, ax=ax[0])
    ax[0].set_title('causal')
    sns.lineplot(data=anti_causal, ax=ax[1])
    ax[1].set_title('anti causal')
    sns.lineplot(data=total, ax=ax[2])
    ax[2].set_title('total')
    sns.lineplot(data=symmetry, ax=ax[3])
    ax[3].set_title('symmetry')
    for p in ax:
        p.set_ylim(-0.05, 1.05)
    plt.pause(1)

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
