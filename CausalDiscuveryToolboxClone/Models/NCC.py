u"""Neural Causation Coefficient.

Author : David Lopez-Paz
Ref :  Lopez-Paz, D. and Nishihara, R. and Chintala, S. and Schölkopf, B. and Bottou, L.,
    "Discovering Causal Signals in Images", CVPR 2017.

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

from sklearn.preprocessing import scale
from scipy.special import expit
import numpy as np
import torch as th
import pandas as pd
# from cdt.causality.pairwise.model import PairwiseModel
from CausalDiscuveryToolboxClone.Models.PairwiseModel import PairwiseModel
from tqdm import trange
from torch.utils import data
import torch.nn as nn
from cdt.utils.Settings import SETTINGS
from utils.symmetry_enforcer import th_enforce_symmetry
from itertools import chain
from scipy.special import expit
import os


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset, labels, device, batch_size=-1):
        'Initialization'
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size if batch_size != 1 else len(dataset)
        self.device = device
        self.nsets = self.__len__() // self.batch_size

    def shuffle(self):
        # self.dataset, self.labels = shuffle(self.dataset, self.labels)
        # z = list(zip(self.dataset, self.labels))
        # print(z)
        # shuffle(z)
        order = th.randperm(len(self.dataset))
        self.dataset = [self.dataset[i] for i in order]
        self.labels = self.labels[order]
        # self.dataset, self.labels = zip(*z)
        if self.device == 'cpu':
            self.set = [
                ([self.dataset[i + j * self.batch_size] for i in range(self.batch_size)],
                 th.index_select(self.labels, 0,
                                 th.LongTensor([i + j * self.batch_size for i in range(self.batch_size)])))
                for j in range(self.nsets)]
        else:
            with th.cuda.device(int(self.device[-1])):
                self.set = [([self.dataset[i + j * self.batch_size]
                              for i in range(self.batch_size)],
                             th.index_select(self.labels, 0,
                                             th.LongTensor([i + j * self.batch_size
                                                            for i in range(self.batch_size)]).cuda()))
                            for j in range(self.nsets)]

    def __iter__(self):
        self.shuffle()
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.nsets:
            self.count += 1
            return self.set[self.count - 1]
        else:
            raise StopIteration

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    # def __getitem__(self, index):
    #     'Generates one sample of data'
    #     # Select sample

    #     # Load data and get label

    #     return self.dataset[index], self.labels[index]


class NCC_model(nn.Module):
    """NCC model structure.

    Args:
        n_hiddens (int): Number of hidden features
        kernel_size (int): Kernel size of the convolutions
    """

    def __init__(self, n_hiddens=100, kernel_size=3, p=0.25, additional_num_hidden_layers=1):
        """Init the NCC structure with the number of hidden units.
        """
        super(NCC_model, self).__init__()
        conv_seq = [
            th.nn.Conv1d(2, n_hiddens, kernel_size),
            th.nn.BatchNorm1d(n_hiddens, affine=False),
            th.nn.ReLU(),
            th.nn.Conv1d(n_hiddens, n_hiddens, kernel_size),
            th.nn.BatchNorm1d(n_hiddens, affine=False),
            th.nn.ReLU()
        ]

        for i in range(additional_num_hidden_layers):
            conv_seq += [
                th.nn.Conv1d(n_hiddens, n_hiddens, kernel_size),
                th.nn.BatchNorm1d(n_hiddens, affine=False),
                th.nn.ReLU()
            ]

        self.conv = th.nn.Sequential(*conv_seq)
        self.conv.apply(self.init_weights)

        dense_seq = []

        for i in range(additional_num_hidden_layers):
            dense_seq += [
                th.nn.Linear(n_hiddens, n_hiddens),
                th.nn.ReLU(),
                th.nn.Dropout(p),
            ]

        dense_seq += [
            th.nn.Linear(n_hiddens, n_hiddens),
            th.nn.ReLU(),
            th.nn.Dropout(p),
            th.nn.Linear(n_hiddens, 1)
        ]

        self.dense = th.nn.Sequential(*dense_seq)
        self.dense.apply(self.init_weights)

    @staticmethod
    def init_weights(m, method='normal'):
        if isinstance(m, th.nn.Linear) or isinstance(m, th.nn.Conv1d):
            if method == 'uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif method == 'normal':
                nn.init.kaiming_normal_(m.weight)
            else:
                raise NotImplemented
            nn.init.normal_(m.bias, 0, 0.001)
        # m.bias.data.fill_(0.01)

    def get_architecture_dict(self):
        architecture_dict = {'encoder': self.conv, 'classifier': self.dense}
        return architecture_dict

    def forward(self, x):
        """Passing data through the network.

        Args:
            x (torch.Tensor): 2d tensor containing both (x,y) Variables

        Returns:
            torch.Tensor: output of NCC
        """

        features = self.conv(x).mean(dim=2)
        return self.dense(features)


class NCC(PairwiseModel):
    u"""Neural Causation Coefficient.

    **Description:** The Neural Causation Coefficient (NCC) is an approach
    neural network relying only on Neural networks to build causally relevant
    embeddings of distributions during training, and classyfing the pairs using
    the last layers of the neural network.

    **Data Type:** Continuous, Categorical, Mixed

    **Assumptions:** This method needs a substantial amount of labelled causal
    pairs to train itself. Its final performance depends on the training set
    used.

    .. note:
        Ref :  Lopez-Paz, D. and Nishihara, R. and Chintala, S. and Schölkopf, B. and Bottou, L.,
        "Discovering Causal Signals in Images", CVPR 2017.

    """

    def __init__(self):
        super(NCC, self).__init__()
        self.model = None
        self.opt = None
        self.criterion = None
        self.anti = True

        self.log_dict = self.create_log_dict()

    @staticmethod
    def create_log_dict_old():
        return {
            'causal':
                {'train': [], 'validation': []},
            'anticausal':
                {'train': [], 'validation': []},
            'total':
                {'train': [], 'validation': []},
            'symmetry':
                {'train': [], 'validation': []},
        }

    @staticmethod
    def create_log_dict():
        return {
            'causal':
                {'train': [], 'validation': []},
            'noncausal':
                {'train': [], 'validation': []},
            'total':
                {'train': [], 'validation': []},
            'symmetry':
                {'train': [], 'validation': []},
        }

    def get_model(self, n_hiddens, kernel_size, dropout_rate, additional_num_hidden_layers):
        self.model = self.model if self.model is not None else NCC_model(n_hiddens, kernel_size, dropout_rate,
                                                                         additional_num_hidden_layers)
        return self.model

    def freeze_weights(self, part=None):
        model = self.model
        architecture_dict = model.get_architecture_dict()
        if part in architecture_dict:
            for param_name, param in architecture_dict[part].named_parameters():
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(f"{name}'s layer weights:\n{param.data}")

    def save_model(self, folder_path, file_path="model.pth"):
        model = self.model
        if model is not None:
            full_path = os.path.join(folder_path, file_path)
            th.save(model.state_dict(), full_path)
        else:
            print('cannot save (no model)')

    def load_model(self, folder_path, file_path):
        full_path = os.path.join(folder_path, file_path)
        if os.path.exists(full_path):
            self.model = NCC_model()
            self.model.load_state_dict(th.load(full_path))
        else:
            print(f"path {full_path} doesn't exist")

    def create_loss(self, learning_rate, optimizer, **kwargs):
        if optimizer.lower() == 'rms':
            self.opt = th.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'adam':
            self.opt = th.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'momentum':
            self.opt = th.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise NotImplemented
        self.criterion = nn.BCEWithLogitsLoss()

    # def fit_clean(self, train_data, train_labels, validation_data, validation_labels, epochs=30, batch_size=16,
    #               verbose=None, device='cpu'):
    #     verbose, device = SETTINGS.get_default(('verbose', verbose), ('device', device))
    #     model = self.model.to(device)
    #     y = th.Tensor(train_labels)
    #     y = y.to(device)
    #     dataset = [th.Tensor(x).t().to(device) for x in train_data]
    #     dat = Dataset(dataset, y, device, batch_size)
    #     data_per_epoch = (len(dataset) // batch_size)
    #
    #     with trange(epochs, desc="Epochs", disable=not verbose) as te:
    #         for _ in te:
    #             self.model.train()
    #             with trange(data_per_epoch, desc="Batches of 2*{}".format(batch_size),
    #                         disable=not (verbose and batch_size == len(dataset))) as t:
    #                 output = []
    #                 labels = []
    #                 for batch, label in dat:
    #                     symmetric_batch, symmetric_label = th_enforce_symmetry(batch, label, self.anti)
    #                     batch += symmetric_batch
    #                     label = th.cat((label, symmetric_label))
    #                     self.opt.zero_grad()
    #                     out = th.stack([model(m.t().unsqueeze(0)) for m in batch], 0).squeeze()
    #                     loss = self.criterion(out, label)
    #                     loss.backward()
    #                     output.append(expit(out.data.cpu()))
    #                     t.set_postfix(loss=loss.item())
    #                     self.opt.step()
    #                     labels.append(label.data.cpu())
    #                 length = th.cat(output, 0).data.cpu().numpy().size
    #                 acc = th.where(th.cat(output, 0).data.cpu() > .5, th.ones((length, 1)).data.cpu(),
    #                                th.zeros((length, 1)).data.cpu()) - \
    #                       th.cat(labels, 0).data.cpu()
    #                 Acc = 1 - acc.abs().mean().item()
    #                 te.set_postfix(Acc=Acc)
    #
    #             self.model.eval()
    #             self.log_values(*self.compute_values(train_data, train_labels, device), 'train')
    #             self.log_values(*self.compute_values(validation_data, validation_labels, device), 'validation')

    def _fit(self, x_tr, y_tr, epochs=50, batch_size=32, learning_rate=0.01, verbose=None, device='cpu', half=True):
        """Fit the NCC model.

        Args:
            x_tr (pd.DataFrame): CEPC format dataframe containing the pairs
            y_tr (pd.DataFrame or np.ndarray): labels associated to the pairs
            epochs (int): number of train epochs
            batch_size (int): size of batch
            learning_rate (float): learning rate of Adam
            verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
            device (str): cuda or cpu device (defaults to ``cdt.SETTINGS.default_device``)
        """

        if half:
            batch_size //= 2
        if batch_size > len(x_tr):
            batch_size = len(x_tr)
        verbose, device = SETTINGS.get_default(('verbose', verbose), ('device', device))
        model = self.model
        # opt = th.optim.Adam(model.parameters(), lr=learning_rate)
        opt = th.optim.RMSprop(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        model = model.to(device)
        y = th.Tensor(y_tr)
        y = y.to(device)
        dataset = [th.Tensor(x).t().to(device) for x in x_tr]
        da = Dataset(dataset, y, device, batch_size)
        data_per_epoch = (len(dataset) // batch_size)

        train_accuracy = []

        with trange(epochs, desc="Epochs", disable=not verbose) as te:
            for _ in te:

                with trange(data_per_epoch, desc="Batches of 2*{}".format(batch_size),
                            disable=not (verbose and batch_size == len(dataset))) as t:
                    output = []
                    labels = []
                    for batch, label in da:
                        # for (batch, label), i in zip(da, t):
                        symmetric_batch, symmetric_label = th_enforce_symmetry(batch, label)
                        batch += symmetric_batch
                        label = th.cat((label, symmetric_label))
                        opt.zero_grad()
                        out = th.stack([model(m.t().unsqueeze(0)) for m in batch], 0).squeeze(2)
                        loss = criterion(out, label)
                        loss.backward()
                        output.append(expit(out.data.cpu()))
                        t.set_postfix(loss=loss.item())
                        opt.step()
                        labels.append(label.data.cpu())
                    length = th.cat(output, 0).data.cpu().numpy().size
                    acc = th.where(th.cat(output, 0).data.cpu() > .5, th.ones((length, 1)).data.cpu(),
                                   th.zeros((length, 1)).data.cpu()) - \
                          th.cat(labels, 0).data.cpu()
                    Acc = 1 - acc.abs().mean().item()
                    te.set_postfix(Acc=Acc)
                    train_accuracy.append(Acc)

    # def fit(self, x_tr, y_tr, epochs=50, batch_size=32, learning_rate=0.01, verbose=None, device='cpu', half=True,
    #         **kwargs):
    #     """Fit the NCC model.
    #
    #     Args:
    #         x_tr (pd.DataFrame): CEPC format dataframe containing the pairs
    #         y_tr (pd.DataFrame or np.ndarray): labels associated to the pairs
    #         epochs (int): number of train epochs
    #         batch_size (int): size of batch
    #         learning_rate (float): learning rate of Adam
    #         verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
    #         device (str): cuda or cpu device (defaults to ``cdt.SETTINGS.default_device``)
    #     """
    #
    #     if half:
    #         batch_size //= 2
    #     if batch_size > len(x_tr):
    #         batch_size = len(x_tr)
    #     verbose, device = SETTINGS.get_default(('verbose', verbose), ('device', device))
    #     model = self.get_model()
    #     opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    #     criterion = nn.BCEWithLogitsLoss()
    #     if kwargs.get('us'):
    #         y = th.Tensor(y_tr)
    #     else:
    #         y = y_tr.values if isinstance(y_tr, pd.DataFrame) else y_tr
    #         y = th.Tensor(y) / 2 + .5
    #     model = model.to(device)
    #     y = y.to(device)
    #     if kwargs.get('us'):
    #         dataset = [th.Tensor(x).t().to(device) for x in x_tr]
    #     else:
    #         dataset = [th.Tensor(np.vstack([row['A'], row['B']])).t().to(device) for (idx, row) in x_tr.iterrows()]
    #     acc_list = [0]
    #
    #     da = Dataset(dataset, y, device, batch_size)
    #     data_per_epoch = (len(dataset) // batch_size)
    #     with trange(epochs, desc="Epochs", disable=not verbose) as te:
    #         for epoch in te:
    #             with trange(data_per_epoch, desc="Batches of 2*{}".format(batch_size),
    #                         disable=not (verbose and batch_size == len(dataset))) as t:
    #                 output = []
    #                 labels = []
    #                 for batch, label in da:
    #                     # for (batch, label), i in zip(da, t):
    #                     symmetric_batch, symmetric_label = th_enforce_symmetry(batch, label)
    #                     batch += symmetric_batch
    #                     label = th.cat((label, symmetric_label))
    #                     opt.zero_grad()
    #                     out = th.stack([model(m.t().unsqueeze(0)) for m in batch], 0).squeeze(2)
    #                     loss = criterion(out, label)
    #                     loss.backward()
    #                     output.append(out)
    #                     t.set_postfix(loss=loss.item())
    #                     opt.step()
    #                     labels.append(label)
    #                 acc = th.where(th.cat(output, 0).data.cpu() > .5, th.ones(len(output)), th.zeros(len(output))) - \
    #                       th.cat(labels, 0).data.cpu()
    #                 te.set_postfix(Acc=1 - acc.abs().mean().item())
    #                 acc_list.append(1 - acc.abs().mean().item())

    def compute_values(self, X, y, device):
        y_val = th.Tensor(y).to(device)
        batch = [th.Tensor(x).t().to(device) for x in X]
        batch_symmetric, symmetric_label = th_enforce_symmetry(batch, y_val, self.anti)
        batch = batch + batch_symmetric
        labels = th.cat((y_val, symmetric_label)).squeeze().data.cpu().numpy()
        logits = self.predict_list(batch)
        output = np.array([expit(logit.item()) for logit in logits])
        preds = np.where(output > .5, np.ones(len(output)), np.zeros(len(output)))
        cause_mask = labels == 0
        err_total_vec = np.abs(preds - labels)
        err_causal = err_total_vec[cause_mask].mean()
        err_non_causal = err_total_vec[~cause_mask].mean()
        err_total = err_total_vec.mean()
        out_reg = output[:len(y)]
        out_sym = output[len(y):]
        symmetry_check = (0.5 * (1 - out_reg + out_sym)).mean() if self.anti else (1 - np.abs(out_sym - out_reg)).mean()
        return err_total, err_causal, err_non_causal, symmetry_check

    def log_values(self, err_total, err_causal, err_anti, symmetry_check, dataset_type):
        assert dataset_type in ['train', 'validation']
        self.log_dict['causal'][dataset_type].append(err_causal)
        self.log_dict['noncausal'][dataset_type].append(err_anti)
        self.log_dict['total'][dataset_type].append(err_total)
        self.log_dict['symmetry'][dataset_type].append(symmetry_check)

    # def train_and_validate(self, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32,
    #                        learning_rate=0.01, verbose=None, device='cpu', half=True):
    #     error_dict = {'causal': {'train': [], 'validation': []},
    #                   'anticausal': {'train': [], 'validation': []},
    #                   'total': {'train': [], 'validation': []}}
    #     symmetry_check_dict = {'train': [], 'validation': []}
    #     self.model = self.get_model()
    #     for epoch in range(epochs):
    #         # self.model.train()
    #         self._fit(X_tr, y_tr, epochs=1, batch_size=batch_size, learning_rate=learning_rate, device=device,
    #                   half=half, verbose=verbose)
    #         self.model.eval()
    #         err_total, err_causal, err_anti, symmetry_check = self.compute_values(X_tr, y_tr, device)
    #         self.log_values(error_dict, symmetry_check_dict, err_total, err_causal, err_anti, symmetry_check, 'train')
    #         err_total, err_causal, err_anti, symmetry_check = self.compute_values(X_val, y_val, device)
    #         self.log_values(error_dict, symmetry_check_dict, err_total, err_causal, err_anti, symmetry_check,
    #                         'validation')
    #
    #     return error_dict, symmetry_check_dict

    def train_old(self, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32, verbose=None, device='cpu', **kwargs):
        self.fit_clean(X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch_size, device=device, verbose=verbose)
        return self.log_dict

    def train(self, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=32, verbose=None, device='cpu', **kwargs):
        verbose, device = SETTINGS.get_default(('verbose', verbose), ('device', device))
        model = self.model.to(device)
        y = th.Tensor(y_tr)
        y = y.to(device)
        dataset = [th.Tensor(x).t().to(device) for x in X_tr]
        dat = Dataset(dataset, y, device, batch_size)
        data_per_epoch = (len(dataset) // batch_size)

        self.model.eval()
        self.log_values(*self.compute_values(X_tr, y_tr, device), 'train')
        self.log_values(*self.compute_values(X_val, y_val, device), 'validation')

        with trange(epochs, desc="Epochs", disable=not verbose) as te:
            for _ in te:
                self.model.train()
                with trange(data_per_epoch, desc="Batches of 2*{}".format(batch_size),
                            disable=not (verbose and batch_size == len(dataset))) as t:
                    output = []
                    labels = []
                    for batch, label in dat:
                        symmetric_batch, symmetric_label = th_enforce_symmetry(batch, label, self.anti)
                        batch += symmetric_batch
                        label = th.cat((label, symmetric_label))
                        self.opt.zero_grad()
                        out = th.stack([model(m.t().unsqueeze(0)) for m in batch], 0).squeeze()
                        loss = self.criterion(out, label)
                        loss.backward()
                        output.append(expit(out.data.cpu()))
                        t.set_postfix(loss=loss.item())
                        self.opt.step()
                        labels.append(label.data.cpu())
                    length = th.cat(output, 0).data.cpu().numpy().size
                    acc = th.where(th.cat(output, 0).data.cpu() > .5, th.ones((length, 1)).data.cpu(),
                                   th.zeros((length, 1)).data.cpu()) - \
                          th.cat(labels, 0).data.cpu()
                    Acc = 1 - acc.abs().mean().item()
                    te.set_postfix(Acc=Acc)

                self.model.eval()
                self.log_values(*self.compute_values(X_tr, y_tr, device), 'train')
                self.log_values(*self.compute_values(X_val, y_val, device), 'validation')
        return self.log_dict

    def get_log_dict(self):
        return self.log_dict

    def _predict_proba(self, X):
        model = self.model
        model.eval()
        return expit(model(th.from_numpy(X)).data.cpu().numpy())

    def _predict(self, batch):
        output = expit(th.stack([self.model(m.t().unsqueeze(0)) for m in batch], 0).squeeze().detach().numpy())
        return output

    def predict_proba(self, dataset, device="cpu", idx=0):
        """Infer causal directions using the trained NCC pairwise model.

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify
            device (str): Device to run the algorithm on (defaults to ``cdt.SETTINGS.default_device``)

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = dataset
        device = SETTINGS.get_default(device=device)
        if self.model is None:
            print('Model has to be trained before doing any predictions')
            raise ValueError
        if len(np.array(a).shape) == 1:
            a = np.array(a).reshape((-1, 1))
            b = np.array(b).reshape((-1, 1))
        m = np.hstack((a, b))
        m = scale(m)
        m = m.astype('float32')
        m = th.from_numpy(m).t().unsqueeze(0)
        m = m.to(device)

        return (self.model(m).data.cpu().numpy() - .5) * 2

    def predict_dataset(self, df, device=None, verbose=None):
        """
        Args:
            x_tr (pd.DataFrame): CEPC format dataframe containing the pairs
            epochs (int): number of train epochs
            learning rate (float): learning rate of Adam
            verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
            device (str): cuda or cpu device (defaults to ``cdt.SETTINGS.default_device``)

        Returns:
            pandas.DataFrame: dataframe containing the predicted causation coefficients
        """
        verbose, device = SETTINGS.get_default(('verbose', verbose),
                                               ('device', device))
        dataset = []
        for i, (idx, row) in enumerate(df.iterrows()):
            a = row['A'].reshape((len(row['A']), 1))
            b = row['B'].reshape((len(row['B']), 1))
            m = np.hstack((a, b))
            m = m.astype('float32')
            m = th.from_numpy(m).t().unsqueeze(0)
            dataset.append(m)

        dataset = [m.to(device) for m in dataset]
        return pd.DataFrame(
            (th.cat([self.model(m) for m, t in zip(dataset, trange(len(dataset)))], 0).data.cpu().numpy() - .5) * 2)

    def predict_list(self, l, device=None, verbose=None):
        """
        Args:
            l (list): CEPC format list containing the pairs
            verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
            device (str): cuda or cpu device (defaults to ``cdt.SETTINGS.default_device``)

        Returns:
            list: list containing the predicted causation coefficients
        """
        verbose, device = SETTINGS.get_default(('verbose', verbose), ('device', device))
        # points = []
        # out = th.stack([self.model(m.t().unsqueeze(0)) for m in l], 0).squeeze()
        # for point in l:
        #     m = np.hstack((a, b))
        # point = point.astype('float32')
        # point = th.from_numpy(point).unsqueeze(0)
        # points.append(point)
        # points = [m.to(device) for m in points]
        # a = [self.model(m.t().unsqueeze(0)) for m in l]
        return [self.model(m.t().unsqueeze(0)) for m in l]
        # return pd.DataFrame(
        #     (th.cat([self.model(m) for m, t in zip(dataset, trange(len(dataset)))], 0).data.cpu().numpy() - .5) * 2)

    def pointification(self, point, device):
        point = point.astype('float32')
        point = th.from_numpy(point).unsqueeze(0)
        return point.to(device)
