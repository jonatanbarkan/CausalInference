import matplotlib.pyplot as plt
import seaborn as sb
from os import path, makedirs, getcwd


def plot_train_val(y_train, y_val, epochs, params, step):
    x = list(range(1, epochs + 1))
    fig, ax = plt.subplots()
    ax.set_xticks(list(range(0, epochs + 1, step)))
    plt.plot(x, y_train, color='b', label='train')
    plt.plot(x, y_val, color='r', label='validation')
    plt.title(params["title"])
    plt.ylabel(params["ylabel"])
    plt.xlabel(params["xlabel"])
    plt.legend(loc='upper right')
    makedirs(params["folder_path"],exist_ok=True)
    full_path = path.join(params["folder_path"], params["file_name"])
    plt.savefig(full_path)
    plt.close()


def make_plots(error_dict, symmetry_check_dict, epochs, model_type='NCC', step=5):
    params = dict()
    params["xlabel"] = "epochs"
    params["folder_path"] = path.join(getcwd(), "Plots", model_type)
    for causal_type in error_dict.keys():
        params["ylabel"] = "validation error"
        params["file_name"] = f'error_{causal_type}.png'
        params["title"] = f"{causal_type} validation error"
        y_train = error_dict[causal_type]['train']
        y_val = error_dict[causal_type]['validation']
        plot_train_val(y_train, y_val, epochs, params, step=step)

    params["ylabel"] = "0.5 (1-NCC(x,y) + NCC(y,x))"
    params["file_name"] = 'symmetry_check.png'
    params["title"] = f"symmetry check"
    y_train = symmetry_check_dict['train']
    y_val = symmetry_check_dict['validation']
    plot_train_val(y_train, y_val, epochs, params, step=step)


