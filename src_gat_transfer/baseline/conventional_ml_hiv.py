__author__ = 'xiangyang'
import sys

sys.path.append('./')

import sys
import tensorflow as tf
import numpy as np
import pickle
import os
import random
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, auc
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
from itertools import chain, combinations
from collections import Counter
from src_convent.utils.utils import get_args, normalize_special_attributes, load_adj_excel, load_adj_csv
from src_convent.utils.config import get_config_from_json, update_config_by_summary


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

REDUCE_GRAPH_FEATURES = False
#REDUCE_GRAPH_FEATURES = True

def cleaning(data):
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
        self.source_psk2index = {}
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
        self.target_psk2index = {}
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
        self.feature_size = 0

    def load(self):
        # load data and build variables
        self.load_graph_features()  # graph features are loaded first so that can be added to attributes
        self.load_attributes()  # individual level features
        if self.is_train:
            self.load_psk2index()
        self.load_mask()

        train_X, test_X, train_Y, test_Y = self.make_dataset()
        print('feature shape:', self.feature_size)
        self.datasize = len(self.features)
        print("datasize: train {}, test {}".format(len(train_Y), len(test_Y)))
        return train_X, test_X, train_Y, test_Y

    def load_attributes(self):
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
                    #print(label)
                    self.labels.append(label)
                    self.features.append(attributes)
                ln += 1

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
        #print(np.shape(self.labels))
        self.labels = np.asarray(self.labels)
        #self.labels = self.np_to_onehot(self.labels, self.config.num_classes)

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
                if REDUCE_GRAPH_FEATURES:
                    self.source_graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.source_graph_features.append([float(row[i]) for i in range(8)])

        print('load target graph features from %s' % self.target_graph_feature_path)
        with open(self.target_graph_feature_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                # reduce means not consider neighbor labels, sometimes reduce perform better
                if REDUCE_GRAPH_FEATURES:
                    self.target_graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.target_graph_features.append([float(row[i]) for i in range(8)])

        self.graph_features = self.source_graph_features + self.target_graph_features

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

    def make_dataset(self):
        # self.features, self.labels, masks
        train_X, test_X, train_Y, test_Y = [], [], [], []
        one = 0
        zero = 0
        for i, _ in enumerate(self.labels):
            mask = self.masks[i]
            #print(mask,self.labels[i])
            if int(mask) == 0:
                zero += 1
                test_Y.append(self.labels[i])
                test_X.append(self.features[i])
            else:
                one += 1
                train_Y.append(self.labels[i])
                train_X.append(self.features[i])
        print("TTTTT",one,zero)
        return train_X, test_X, train_Y, test_Y

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


def train_lr(config,source_city='houston'):
    source_feature_path = config.exp_dir + config.source_ind_feature_path
    source_train_mask_path = config.exp_dir + config.source_train_mask_path
    source_graph_feature_path = config.exp_dir + config.source_graph_feature_path
    source_psk2index_path = config.exp_dir + config.source_psk2index_path

    target_feature_path = config.exp_dir + config.target_ind_feature_path
    target_train_mask_path = config.exp_dir + config.target_train_mask_path
    target_graph_feature_path = config.exp_dir + config.target_graph_feature_path
    target_psk2index_path = config.exp_dir + config.target_psk2index_path

    if source_city == 'houston':

        source_sex_adj_path = config.exp_dir + config.houston_sex_adj_path
        source_venue_adj_path = config.exp_dir + config.houston_venue_adj_path

        target_sex_adj_path = config.exp_dir + config.chicago_sex_adj_path
        target_venue_adj_path = config.exp_dir + config.chicago_sex_adj_path

    else:
        source_sex_adj_path = config.exp_dir + config.chicago_sex_adj_path
        source_venue_adj_path = config.exp_dir + config.chicago_sex_adj_path

        target_sex_adj_path = config.exp_dir + config.houston_sex_adj_path
        target_venue_adj_path = config.exp_dir + config.houston_venue_adj_path

    is_train = True
    loader = PatientLoader(config,source_feature_path,source_sex_adj_path, source_venue_adj_path,
                 source_train_mask_path, source_graph_feature_path, source_psk2index_path,
                 target_feature_path, target_sex_adj_path, target_venue_adj_path, target_train_mask_path,
                 target_graph_feature_path, target_psk2index_path, is_train)
    trainX, testX, trainY, testY = loader.load()
    headers = selected_attributes
    
    trainY = np.array(trainY)
    testY = np.array(testY)
    '''
    trainY = trainY.flatten()[:len(trainX)]
    testY = testY.flatten()[len(trainX):]
    '''
    # Using SMOTE
    print("xxx", trainY.shape,testY.shape)
    # print('Original dataset shape {}'.format(Counter(trainX)))
    # sm = SMOTE(random_state=0)
    # print(trainX.shape, trainY.shape)
    # trainX, trainY = sm.fit_sample(trainX, trainY)
    # print('Resampled dataset shape {}'.format(Counter(trainX)))

    for train_percent in [1.0]:#, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05]:
        max_auc = 0.5
        train_size = int(len(trainY) * train_percent)
        temp_trainX = trainX[:train_size]
        temp_trainY = trainY[:train_size]
        print("train on {} samples and test on {} samples".format(len(temp_trainY), len(testY)))
        paras = ""
        for i, C in enumerate((0.0001, 0.001, 0.01, 1, 10, 100)):
            for penalty in ['l2']:
                classifier = linear_model.LogisticRegression(C=C, penalty=penalty, tol=0.01,max_iter=10000)

                classifier.fit(temp_trainX, temp_trainY)
                preds = classifier.predict(testX)
                probs = classifier.predict_proba(testX)
                probs = probs[:, 1]
                # print(probs.shape)
                # input()
                print(np.shape(testY),np.shape(probs))
                auc_score = roc_auc_score(testY, probs)

                # rankings = [[headers[i], classifier.coef_[0][i]] for i in range(len(headers))]
                # rankings = sorted(rankings, key=lambda x: x[1], reverse=True)
                # print('\t', rankings[:10])
                # input()
                # print ("\tAuc on testing for C={} tol=0.01 for L1 penalty is {}".format(C, auc_score))

                if auc_score > max_auc:
                    max_auc = auc_score
                    paras = "penalty %s and C %f" % (penalty, C)

        print ('\t#LR: max AUC for train percent {} is: {}, with para {}\n'.format(train_percent, max_auc, paras))


def train_rf(config,source_city='houston'):
    source_feature_path = config.exp_dir + config.source_ind_feature_path
    source_train_mask_path = config.exp_dir + config.source_train_mask_path
    source_graph_feature_path = config.exp_dir + config.source_graph_feature_path
    source_psk2index_path = config.exp_dir + config.source_psk2index_path

    target_feature_path = config.exp_dir + config.target_ind_feature_path
    target_train_mask_path = config.exp_dir + config.target_train_mask_path
    target_graph_feature_path = config.exp_dir + config.target_graph_feature_path
    target_psk2index_path = config.exp_dir + config.target_psk2index_path

    if source_city == 'houston':

        source_sex_adj_path = config.exp_dir + config.houston_sex_adj_path
        source_venue_adj_path = config.exp_dir + config.houston_venue_adj_path

        target_sex_adj_path = config.exp_dir + config.chicago_sex_adj_path
        target_venue_adj_path = config.exp_dir + config.chicago_sex_adj_path

    else:
        source_sex_adj_path = config.exp_dir + config.chicago_sex_adj_path
        source_venue_adj_path = config.exp_dir + config.chicago_sex_adj_path

        target_sex_adj_path = config.exp_dir + config.houston_sex_adj_path
        target_venue_adj_path = config.exp_dir + config.houston_venue_adj_path
    
    loader = PatientLoader(config,source_feature_path,source_sex_adj_path, source_venue_adj_path,
                 source_train_mask_path, source_graph_feature_path, source_psk2index_path,
                 target_feature_path, target_sex_adj_path, target_venue_adj_path, target_train_mask_path,
                 target_graph_feature_path, target_psk2index_path, is_train=True)
    trainX, testX, trainY, testY = loader.load()
    headers = selected_attributes
    trainY = np.array(trainY)
    testY = np.array(testY)
    '''
    trainY = trainY.flatten()[:len(trainX)]
    testY = testY.flatten()[len(trainX):]
    '''
    # Using SMOTE
    # print(trainX.shape, trainY.shape)
    # print('Original dataset shape {}'.format(Counter(trainX)))
    # sm = SMOTE(random_state=0)
    # print(trainX.shape, trainY.shape)
    # trainX, trainY = sm.fit_sample(trainX, trainY)
    # print('Resampled dataset shape {}'.format(Counter(trainX)))

    for train_percent in [1.0]:#, 0.8, 0.5, 0.3, 0.1]:#, 0.08, 0.05]:
        max_auc = 0.5
        train_size = int(len(trainY) * train_percent)
        temp_trainX = trainX[:train_size]
        temp_trainY = trainY[:train_size]
        #print(np.shape(trainY),np.shape(temp_trainY))
        print("train on {} samples and test on {} samples".format(len(temp_trainY), len(testY)))
        paras = ""
        for d in [None,1,2,3,10,50]:
            for s in [0,2,3,10]:
                classifier = RandomForestClassifier(max_depth=d, random_state=s)
                classifier.fit(temp_trainX, temp_trainY)
                preds = classifier.predict(testX)
                probs = classifier.predict_proba(testX)
                # print(np.asarray(testX).shape, np.asarray(probs).shape)
                # input()
                probs = probs[:, 1]
                # print(probs.shape)
                # input()
                auc_score = roc_auc_score(testY, probs)

                importances = classifier.feature_importances_
                std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                # print('AUC %.3f' % auc_score)
                # print("Feature ranking:")
                #
                # for f in range(10):#np.asarray(temp_trainX).shape[1]):
                #     print("%d. feature %s (%f)" % (f + 1, headers[indices[f]], importances[indices[f]]))
                # input()

                if auc_score > max_auc:
                    max_auc = auc_score
                    paras = "max_depth %s and random state %d" % (str(d), s)
        print ('\t#RF: max AUC for train percent {} is: {}, with parameters {}\n'.format(train_percent, max_auc, paras))

if __name__ == "__main__":
    random.seed(2010)
    args = get_args()
    print("getting config from {}".format(args.config))
    config, _ = get_config_from_json(args.config)
    config = update_config_by_summary(config)  # add summary and model directory
    train_lr(config)
    train_rf(config)
