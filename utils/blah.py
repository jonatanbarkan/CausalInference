import numpy as np
import torch as th
import torchvision


def compare_models(model1, model2):
    d = dict()
    for (name1, param1) in model1.model.named_parameters():
        for (name2, param2) in model2.model.named_parameters():
            if name1 == name2:
                d[name1] = (param1.data == param2.data).min()
    print(d)
    return d
