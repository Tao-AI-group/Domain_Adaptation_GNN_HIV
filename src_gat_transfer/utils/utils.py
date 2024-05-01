import argparse
import json
import csv
from xlrd import open_workbook
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--delete',
        metavar='D',
        default='0',
        help='To delete the previous checkpoints and summaries'
    )
    argparser.add_argument(
        '-t', '--threshold',
        metavar='T',
        default='0',
        help='Specify the threshold for defining the venue co-affiliation matrix: e.g. if the value > threshold, an edge exists.'
    )

    args = argparser.parse_args()
    return args


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_adj_excel(workbook_path, psk2index):
    """
    need self loop and symmetric
    :param workbook_path:
    :param psk2index:
    :return:
    """
    patient_size = len(psk2index)
    adj_matrix = np.zeros(shape=(patient_size, patient_size), dtype=int)
    # add self loop
    # for i in range(patient_size):
    #     adj_matrix[i][i] = 1

    workbook = open_workbook(workbook_path)
    w1_sheet = workbook.sheet_by_index(0)  # sheet1
    # the header
    header = [str(w1_sheet.cell(0, i).value) for i in range(1, w1_sheet.ncols)]  # attributes

    for row_idx in range(1, w1_sheet.nrows):
        row = [w1_sheet.cell(row_idx, i).value for i in range(w1_sheet.ncols)]
        head, tail = psk2index[int(row[0])], psk2index[int(row[1])]
        adj_matrix[head][tail] = 1

    # symmetric
    for i in range(patient_size):
        for j in range(patient_size):
            if adj_matrix[i][j] == 1:
                adj_matrix[j][i] = 1
    # display_adj(adj_matrix)
    return adj_matrix


def load_adj_csv(csv_path, psk2index):
    patient_size = len(psk2index)
    adj_matrix = np.zeros(shape=(patient_size, patient_size), dtype=int)

    # add self loop
    # for i in range(patient_size):
    #     adj_matrix[i][i] = 1

    with open(csv_path) as ifile:
        ln = 0
        patient_venue_matrix = []
        psks = []
        for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if ln == 0:
                row = row
                continue
            else:
                head, tail = psk2index[int(row[0])], psk2index[int(row[1])]
                adj_matrix[head][tail] = 1

    # symmetric
    for i in range(patient_size):
        for j in range(patient_size):
            if adj_matrix[i][j] == 1:
                adj_matrix[j][i] = 1
    # display_adj(adj_matrix)
    return adj_matrix


def normalize_special_attributes(attributes):
    """
    add a smooth factor to NULL values, replace > and <
    :param attributes:
    :return:
    """
    epsilon = 1e-12
    for index, att in enumerate(attributes):
        # avoid empty values
        if len(str(att)) == 0:
            attributes[index] = epsilon

        # avoid 0s in log and sqrt
        if 'vl_w1' == att or 'vl_w2' == att:
            if '<' in attributes[index] or '>' in attributes[index]:
                attributes[index] = attributes[index][1:]
            attributes[index] = np.log(attributes[index] + epsilon)

        elif 'num_sex_partner_w1' == att or 'num_sex_partner_w2' == att:
            attributes[index] = np.sqrt(attributes[index] + epsilon)
    return attributes


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) # row sum of each document as the summary of all words
    r_inv = np.power(rowsum, -1).flatten()  # inverse
    r_inv[np.isinf(r_inv)] = 0. # if infinity then bzero
    r_mat_inv = sp.diags(r_inv) # convert to diagonal matrix, numbers on the central diagonal
    features = r_mat_inv.dot(features) # normalize by the row sum
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    print('nb graphs:', nb_graphs) # when to include more graphs?
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1]))) # inside: add self-loop, outside: no change if nhood=1,
            # if nhood>1, seems no change either if reduce to 1

        for i in range(sizes[g]): # reduce to non weighted edge
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def f1(y_true, y_preds):
    return f1_score(y_true, y_preds)


def prf(y_true, y_preds):
    p,r,f,s = precision_recall_fscore_support(y_true, y_preds, average='macro')
    return p, r, f


def auc_value(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)
    return auc(fpr, tpr)

def auprc_value(y_true, probs):
    return average_precision_score(y_true, probs)

def cmatrix(y_true, y_preds):
    return confusion_matrix(y_true, y_preds)
