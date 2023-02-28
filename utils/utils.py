import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re

def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    f = open("data/ind.{}.adj".format(dataset_str), 'rb')
    adj = pkl.load(f, encoding='latin1')
    print(adj.shape)

    f = open("data/ind.{}.allx".format(dataset_str), 'rb')
    allx = pkl.load(f, encoding='latin1')
    print(adj.shape)

    # x, y, tx, ty, allx, ally, adj = tuple(objects)
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # features = sp.vstack((allx, tx)).tolil()
    # labels = np.vstack((ally, ty))
    # print(len(labels))

    # train_idx_orig = parse_index_file(
    #     "data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_orig)

    # val_size = train_size - x.shape[0]
    # test_size = tx.shape[0]

    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + val_size)
    # idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, allx
    # , features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

if __name__ == '__main__':
    adj, allx = load_corpus('res14')
    import pdb; pdb.set_trace()