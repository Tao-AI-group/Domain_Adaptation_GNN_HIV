"""
data loader for source+target network
    left top quarter: source
    right bottom quarter: target
"""
import sys

sys.path.append('/ctao_shared_data/HIV/aid_social_venue_hiv_prediction/')

import logging
from pathlib import Path
import string
import numpy as np
import random
import pickle as pkl
import csv
from scipy import sparse
from xlrd import open_workbook
# import tensorflow as tf
from src_gat_transfer.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gat_transfer.utils.dirs import create_dirs
from src_gat_transfer.utils.logger import Logger
from src_gat_transfer.utils.utils import get_args, load_adj_csv, load_adj_excel, adj_to_bias, \
    sparse_to_tuple, normalize_adj

random.seed(2010)

REDUCE_GRAPH_FEATURES = True
ABLATION = False

# PrEP and viral load relevant features are extremely associated with the determination of hiv so we remove them
# Features considered:
#       node: socio-demographic, drug usage, sex behavior, depression
#       graph features: known hiv positive neighbors, number of social venues attended, number of health venues attended
selected_attributes = [
                       'race',
                       'age_w1',
                       'black',
                       'hispanicity',
                       'smallnet_w1',
                       'education_w1',
                       'sexual_identity_w1',
                       'past12m_homeless_w1',
                       'insurance_type_w1',
                       'inconsistent_condom_w1',
                       'ever_jailed_w1',
                       'age_jailed_w1',
                       'freq_3mo_tobacco_w1',
                       'freq_3mo_alcohol_w1',
                       'freq_3mo_cannabis_w1',
                       'freq_3mo_inhalants_w1',
                       'freq_3mo_hallucinogens_w1',
                       'freq_3mo_stimulants_w1',
                       'ever_3mo_depressants_w1',
                       'num_sex_partner_drugs_w1',
                       'num_nom_sex_w1',
                       'num_nom_soc_w1',
                       'num_sex_partner_w1',
                       'num_oral_partners_w1',
                       'num_anal_partners_w1',
                       'sex_transact_money_w1',
                       'sex_transact_others_w1',
                       'depression_sum_w1',
                       ]
INTER_DEFAULT = -100


def cleaning(data):
    """ data cleaning for None values """
    for i, x in enumerate(data):
        for j, elem in enumerate(x):
            if np.isnan(elem) or elem == 'None':
                data[i][j] = INTER_DEFAULT
    return data


class PatientLoader:

    def __init__(self, config, source_feature_path, source_sex_adj_path, source_venue_adj_path,
                                 source_train_mask_path, source_graph_feature_path, source_psk2index_path,
                                 target_feature_path, target_sex_adj_path, target_venue_adj_path, target_train_mask_path,
                                 target_graph_feature_path, target_psk2index_path, is_train):
        self.config = config
        # source
        self.source_feature_path = source_feature_path
        self.source_sex_adj_path = source_sex_adj_path
        self.source_venue_adj_path = source_venue_adj_path
        self.source_mask_path = source_train_mask_path
        self.source_graph_feature_path = source_graph_feature_path
        self.source_psk2index_path = source_psk2index_path
        self.source_labels = []  # Y
        self.source_features = []  # X
        self.source_graph_features = []  # X extension
        self.source_indices = []
        self.source_masks = []
        self.source_sex_biases = []  # for GAT
        self.source_adj_sex = []
        self.source_venue_biases = []  # for GAT
        self.source_adj_venue = []
        self.source_psk2index = {} # the only dict
        self.source_dataset = []
        self.source_datasize = 0
        # target
        self.target_feature_path = target_feature_path
        self.target_sex_adj_path = target_sex_adj_path
        self.target_venue_adj_path = target_venue_adj_path
        self.target_mask_path = target_train_mask_path
        self.target_graph_feature_path = target_graph_feature_path
        self.target_psk2index_path = target_psk2index_path
        self.target_labels = []  # Y
        self.target_features = []  # X
        self.target_graph_features = []  # X extension
        self.target_indices = []
        self.target_masks = []
        self.target_sex_biases = []  # for GAT
        self.target_adj_sex = []
        self.target_venue_biases = []  # for GAT
        self.target_adj_venue = []
        self.target_psk2index = {}   # the only dict
        self.target_dataset = []
        self.target_datasize = 0
        
        self.feature_size = 0
        self.is_train = is_train

        """
        the final graph
            self.support -> for GCN
            self.adj -> for GAT
            self.biases -> for GAT
        """
        self.support = []
        self.adj = []
        self.biases = []
        self.labels = []
        self.features = []
        self.masks = []

    def load(self):
        # load data and build variables
        self.load_graph_features()  # graph features are loaded first so that can be added to attributes
        self.load_attributes()   # individual level features
        if self.is_train:
            self.load_psk2index()
            self.load_adj()
        self.load_mask()
        self.mask_labels()

        # standard operations of preprocessing for GAT, add another dimension
        self.features = self.features[np.newaxis]
        self.labels = self.labels[np.newaxis]
        self.masks = self.masks[np.newaxis]

        self.dataset = list(zip(self.features, self.labels, self.masks))
        print('num of features:', self.features.shape)
        self.datasize = self.features.shape[0]
        print("num of samples:", self.datasize)

    def load_attributes(self):
        # load source attributes
        attr_indices = []
        with open(self.source_feature_path) as ifile:
            ln = 0
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    header = row
                    attribute2index = {k: i for i, k in enumerate(header)}
                    attr_indices = [attribute2index[a] for a in selected_attributes]
                else:
                    # for hiv label
                    index, label_w1, label_w2, attributes = int(row[0]), int(row[1]), int(row[3]), row
                    attributes = [attributes[i] for i in attr_indices]
                    # normalize again some attributes to be used
                    attributes = self.recode_attributes(attributes)
                    attributes = [float(a) if a != "" and a != "NA" else -100.0 for a in attributes]

                    # add graph features
                    graph_attributes = self.source_graph_features[ln - 1]  # ln=0 is the header, so start from 1
                    attributes.extend(graph_attributes)

                    # allow diagnosis of HIV to be lagging
                    label = self.make_label_from_two_wave(label_w1, label_w2)
                    self.labels.append(label)
                    self.features.append(attributes)
                ln += 1

        # load target attributes
        attr_indices = []
        with open(self.target_feature_path) as ifile:
            ln = 0
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    header = row
                    attribute2index = {k: i for i, k in enumerate(header)}
                    attr_indices = [attribute2index[a] for a in selected_attributes]
                else:
                    # for hiv label
                    index, label_w1, label_w2, attributes = int(row[0]), int(row[1]), int(row[3]), row
                    attributes = [attributes[i] for i in attr_indices]
                    # normalize again some attributes to be used
                    attributes = self.recode_attributes(attributes)
                    attributes = [float(a) if a != "" and a != "NA" else -100.0 for a in attributes]

                    # add graph features
                    graph_attributes = self.target_graph_features[ln - 1]  # ln=0 is the header, so start from 1
                    attributes.extend(graph_attributes)

                    # allow diagnosis of HIV to be lagging
                    label = self.make_label_from_two_wave(label_w1, label_w2)
                    self.labels.append(label)
                    self.features.append(attributes)
                ln += 1

        # join source and target into a big network, so label and feature can be only one variable
        self.labels = np.asarray(self.labels)
        self.labels = self.np_to_onehot(self.labels, self.config.num_classes)

        self.features = np.asarray(self.features, dtype=float)
        self.feature_size = self.features.shape[1]

    def load_mask(self):
        print('load source mask from %s' % self.source_mask_path)
        with open(self.source_mask_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                self.source_masks.append(int(row[0]))

        print('load target mask from %s' % self.target_mask_path)
        with open(self.target_mask_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                self.target_masks.append(int(row[0]))

        # since only one network is needed, we combine source and target together
        self.masks = self.source_masks + self.target_masks
        self.masks = np.asarray(self.masks)

    def load_graph_features(self):
        print('load source graph features from %s' % self.source_graph_feature_path)
        # 0.sex_centrality,     1.venue_centrality,  2.sex_num_neighbor,  3.venue_num_neighbor,
        # 4.num_social_venues,  5.num_health_venues, 6.hiv_pos_ratio,     7.hiv_neg_ratio,
        # 8.syphilis_pos_ratio, 9.syphilis_neg_ratio
        with open(self.source_graph_feature_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                # reduce means not consider neighbor labels, sometimes reduce perform better
                if ABLATION:
                    self.source_graph_features.append([float(row[i]) for i in range(6)])
                elif REDUCE_GRAPH_FEATURES:
                    self.source_graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.source_graph_features.append([float(row[i]) for i in range(8)])

        print('load target graph features from %s' % self.target_graph_feature_path)
        with open(self.target_graph_feature_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                # reduce means not consider neighbor labels, sometimes reduce perform better
                if ABLATION:
                    self.target_graph_features.append([float(row[i]) for i in range(6)])
                elif REDUCE_GRAPH_FEATURES:
                    self.target_graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.target_graph_features.append([float(row[i]) for i in range(8)])

        # since only one network is needed, we combine source and target together
        self.graph_features = self.source_graph_features + self.target_graph_features
        print('graph_features size', len(self.graph_features))

    def load_psk2index(self):
        ln = 0
        with open(self.source_psk2index_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    ln += 1
                else:
                    psk, index = row[0], row[1]
                    self.source_psk2index[int(psk)] = int(index)

        ln = 0
        with open(self.target_psk2index_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    ln += 1
                else:
                    psk, index = row[0], row[1]
                    self.target_psk2index[int(psk)] = int(index)

    def load_adj(self):
        '''
        if '.csv' in self.sex_adj_path:
            self.adj_sex = load_adj_csv(self.sex_adj_path, self.psk2index)
        elif '.xls' in self.sex_adj_path:
            self.adj_sex = load_adj_excel(self.sex_adj_path, self.psk2index)
        '''
        self.source_adj_sex, _ = pkl.load(open(self.source_sex_adj_path, "rb"))
        # social w1, health w1, social w2, health w2, [As]
        self.source_adj_venue, _, _, _, patient2venue, _, _, _ = pkl.load(open(self.source_venue_adj_path, "rb"))
        self.source_adj_venue = self.stratify_venue_matrix(self.source_adj_venue, self.config.venue_thres)
        
        self.target_adj_sex, _ = pkl.load(open(self.target_sex_adj_path, "rb"))
        # social w1, health w1, social w2, health w2, [As]
        self.target_adj_venue, _, _, _, patient2venue, _, _, _ = pkl.load(open(self.target_venue_adj_path, "rb"))
        self.target_adj_venue = self.stratify_venue_matrix(self.target_adj_venue, self.config.venue_thres)

        source_num_nodes = self.config.houston_num_nodes if self.config.source_city == 'houston' \
            else self.config.chicago_num_nodes
        target_num_nodes = self.config.houston_num_nodes if self.config.source_city == 'chicago' \
            else self.config.chicago_num_nodes
        num_nodes = source_num_nodes + target_num_nodes

        """ build big graph """
        supra_graph = np.zeros((num_nodes, num_nodes), dtype=int)

        # change to sex/venue for further exp if needed
        supra_graph[:source_num_nodes, :source_num_nodes] = self.source_adj_sex #source_adj_venue/sex 
        supra_graph[source_num_nodes:, source_num_nodes:] = self.target_adj_sex # target_adj_venue/sex

        ## for GCN
        if self.config.model_version == 'gcn':
            # dense to sparse
            support = sparse.csr_matrix(supra_graph)
            # normalize
            support = normalize_adj(support + sparse.eye(support.shape[0]))
            # to tuple
            self.support = sparse_to_tuple(support)

        ## for GAT
        if self.config.model_version == 'gat':
            self.adj = supra_graph[np.newaxis]
            self.biases = adj_to_bias(self.adj, [num_nodes], nhood=1)

        print('adjacent matrix shape:', self.adj.shape)

    def get_datasize(self):
        return self.datasize

    def get_dataset(self):
        return self.dataset

    def get_feature_size(self):
        return self.feature_size

    def next_batch(self, prev_idx):
        """
        the next batch of data for training
        :param prev_idx:
        :return:
        """
        b = self.config.batch_size
        upper = np.min([self.datasize, b * (prev_idx + 1)], axis=0)
        yield self.dataset[b * prev_idx: upper]

    def mask_labels(self):
        for i, _ in enumerate(self.labels):
            mask = self.masks[i]
            if int(mask) == 0:
                self.labels[i] = np.asarray([0]*self.config.num_classes)

    @staticmethod
    def np_to_onehot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    @staticmethod
    def make_label_from_two_wave(label_w1, label_w2):
        return label_w1 or label_w2

    @staticmethod
    def recode_attributes(attributes):
        new_attributes = []
        for i, name in enumerate(selected_attributes):
            value = attributes[i]
            if value == 'NA' or value == '':
                new_attributes.append(value)
                continue
            if 'sexual_identity' in name:
                if float(value) == 1 or float(value) == 3:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif name == 'education':
                if float(value) <= 2:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif 'age_jailed' in name:
                if value == '12 or younger':
                    new_attributes.append(12.0)
                else:
                    new_attributes.append(value)
            else:
                new_attributes.append(value)
        return new_attributes

    @staticmethod
    def stratify_venue_matrix(matrix, thres):
        new_matrix = []
        for row in matrix:
            new_matrix.append([1 if i >= thres else 0 for i in row])
        return np.asarray(new_matrix)
