import networkx as nx
import numpy as np


def get_centrality(node, adj):
    """
    get centrality for a certain node, centrality: ?
    :param node:
    :param adj:
    :return:
    """
    G = nx.from_numpy_matrix(adj)
    centrality = nx.eigenvector_centrality(G, max_iter=250)
    #print(G.size,len(adj))
    return centrality[node]


def get_num_neighbors(node, adj):
    """
    get number of neighbors
    :param node:
    :param adj:
    :return:
    """
    neighbors = adj[node]
    neighbors = [1 if n > 0 else 0 for n in neighbors] # consider that in the venue network, there might be over 1
    row_sum = np.sum(np.asarray(neighbors))  # no isolated node
    total = adj.shape[0]
    return row_sum / total


def get_num_venues(node, patient2venue_matrix):
    venues = patient2venue_matrix[node]
    venues = [1 if venue > 0 else 0 for venue in venues]
    row_sum = np.sum(np.asarray(venues))
    total = patient2venue_matrix.shape[0]
    return row_sum / total


def get_polarity_ratio(node, adj, labels, train_masks):
    """
    get #positive and #negative for a certain node from known training set
    """
    num_pos, num_neg = 0.0, 0.0
    neighbors = adj[node]
    row_sum = np.sum(np.asarray(neighbors)) # there are isolated node
    if row_sum == 0:
        return row_sum, 0.0, 0.0

    # nodes with known labels
    train_label_map = {i:labels[i] for i in range(len(train_masks)) if int(train_masks[i]) == 1}

    for i, n in enumerate(neighbors):
        if n == 1:
            if i in train_label_map:
                if train_label_map[i] == 0:
                    num_neg += 1
                else:
                    num_pos += 1
                # print(i, num_pos, num_neg)

    return row_sum, num_pos/row_sum, num_neg/row_sum
