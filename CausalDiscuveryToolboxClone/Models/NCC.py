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
import numpy as np
import torch as th
import pandas as pd
# from cdt.causality.pairwise.model import PairwiseModel
from CausalDiscuveryToolboxClone.Models.PairwiseModel import PairwiseModel
from tqdm import trange
from torch.utils import data
from cdt.utils.Settings import SETTINGS
from utils.symmetry_enforcer import th_enforce_symmetry


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


class NCC_model(th.nn.Module):
    """NCC model structure.

    Args:
        n_hiddens (int): Number of hidden features
        kernel_size (int): Kernel size of the convolutions
    """

    def __init__(self, n_hiddens=100, kernel_size=3, p=0.25):
        """Init the NCC structure with the number of hidden units.
        """
        super(NCC_model, self).__init__()
        self.conv = th.nn.Sequential(th.nn.Conv1d(2, n_hiddens, kernel_size),
                                     th.nn.BatchNorm1d(n_hiddens, affine=False),
                                     th.nn.ReLU(),
                                     th.nn.Conv1d(n_hiddens, n_hiddens,
                                                  kernel_size),
                                     th.nn.BatchNorm1d(n_hiddens, affine=False),
                                     th.nn.ReLU())
        # self.batch_norm = th.nn.BatchNorm1d(n_hiddens, affine=False)
        self.dense = th.nn.Sequential(th.nn.Linear(n_hiddens, n_hiddens),
                                      # th.nn.BatchNorm1d(n_hiddens, affine=False),
                                      th.nn.ReLU(),
                                      th.nn.Dropout(p),
                                      th.nn.Linear(n_hiddens, 1)
                                      )

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

    def fit(self, x_tr, y_tr, epochs=50, batch_size=32, learning_rate=0.01, verbose=None, device='cpu', half=True,
            **kwargs):
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
        self.model = NCC_model()
        opt = th.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = th.nn.BCEWithLogitsLoss()
        if kwargs.get('us'):
            y = th.Tensor(y_tr)
        else:
            y = y_tr.values if isinstance(y_tr, pd.DataFrame) else y_tr
        y = th.Tensor(y) / 2 + .5
        self.model = self.model.to(device)
        y = y.to(device)
        if kwargs.get('us'):
            dataset = [th.Tensor(x).t().to(device) for x in x_tr]
        else:
            dataset = [th.Tensor(np.vstack([row['A'], row['B']])).t().to(device) for (idx, row) in x_tr.iterrows()]
        acc_list = [0]
        da = Dataset(dataset, y, device, batch_size)
        data_per_epoch = (len(dataset) // batch_size)
        with trange(epochs, desc="Epochs", disable=not verbose) as te:
            for epoch in te:
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
                        out = th.stack([self.model(m.t().unsqueeze(0)) for m in batch], 0).squeeze(2)
                        loss = criterion(out, label)
                        loss.backward()
                        output.append(out)
                        t.set_postfix(loss=loss.item())
                        opt.step()
                        labels.append(label)
                    acc = th.where(th.cat(output, 0).data.cpu() > .5, th.ones(len(output)), th.zeros(len(output))) - \
                          th.cat(labels, 0).data.cpu()
                    te.set_postfix(Acc=1 - acc.abs().mean().item())
                    acc_list.append(1 - acc.abs().mean().item())

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
        dataset = []
        for point in l:
            # m = np.hstack((a, b))
            point = point.astype('float32')
            point = th.from_numpy(point).unsqueeze(0)
            dataset.append(point)

        dataset = [m.to(device) for m in dataset]
        return pd.DataFrame(
            (th.cat([self.model(m) for m, t in zip(dataset, trange(len(dataset)))], 0).data.cpu().numpy() - .5) * 2)
