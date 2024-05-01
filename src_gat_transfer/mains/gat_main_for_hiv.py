"""
load all source data (X, Y, adj), load all target data with masks to distinguish train and test
combine the two networks into a larger one like supra graph
consider source and target are two non-adjacent graphs
"""
import sys

sys.path.append('./')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
from src_gat_transfer.data_loader.patient_loader_for_hiv import PatientLoader
from src_gat_transfer.models.gat_transfer import GAT
from src_gat_transfer.trainers.gat_transfer_trainer import GraphTrainer as GatTrainer
from src_gat_transfer.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gat_transfer.utils.dirs import create_dirs
from src_gat_transfer.utils.logger import Logger
from src_gat_transfer.utils.utils import get_args
from pathlib import Path
import shutil
import pickle as pkl

import tensorflow as tf

tf.compat.v1.random.set_random_seed(1234)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    print("getting config from {}".format(args.config))
    config, _ = get_config_from_json(args.config)
    config = update_config_by_summary(config)  # add summary and model directory
    # if remove the previous results, set -d 1
    print("If delete previous checkpoints {}".format(args.delete))
    if args.delete == 'xx':
        # delete existing model and summaries
        print('Deleting existing models and logs from:')
        # best_model_dir is under model dir
        print(config.summary_dir, config.model_dir, config.best_model_dir)
        path = Path(config.summary_dir)
        shutil.rmtree(path)
        path = Path(config.model_dir)
        shutil.rmtree(path)
        path = Path(config.best_model_dir)
        shutil.rmtree(path)

    config.venue_thres = int(args.threshold)

    # create the experiments dirs
    # summary dir, model dir defined in json ?
    create_dirs([config.summary_dir, config.model_dir, config.best_model_dir])

    # create your data generator to load train data
    print("Training using {}".format(config.model_version))

    # fix the model and trainer for experiment
    Model = GAT
    Trainer = GatTrainer
    source_path = config.exp_dir
    target_path = config.exp_dir
    if 'houston' in target_path:
        target_path = target_path.replace('houston','chicago')
    elif 'chicago' in target_path:
        target_path = target_path.replace('chicago','houston')

    source_feature_path = config.exp_dir + config.source_ind_feature_path
    source_train_mask_path = config.exp_dir + config.source_train_mask_path
    source_test_mask_path = config.exp_dir + config.source_test_mask_path
    target_feature_path = target_path + config.target_ind_feature_path
    target_train_mask_path = target_path + config.target_train_mask_path
    target_test_mask_path = target_path + config.target_test_mask_path

    if config.source_city == 'houston':
        source_sex_adj_path = config.houston_sex_adj_path
        source_venue_adj_path = config.houston_venue_adj_path
        target_sex_adj_path = config.chicago_sex_adj_path
        target_venue_adj_path = config.chicago_venue_adj_path
    else:
        source_sex_adj_path = config.chicago_sex_adj_path
        source_venue_adj_path = config.chicago_venue_adj_path
        target_sex_adj_path = config.houston_sex_adj_path
        target_venue_adj_path = config.houston_venue_adj_path


    source_graph_feature_path = config.exp_dir + config.source_graph_feature_path
    source_psk2index_path = config.exp_dir + config.source_psk2index_path
    target_graph_feature_path = target_path + config.target_graph_feature_path
    target_psk2index_path = target_path + config.target_psk2index_path

    # 10 random realizations of train-test split and average, no valid is needed
    train_loader = PatientLoader(config, source_feature_path, source_sex_adj_path, source_venue_adj_path,
                                 source_train_mask_path, source_graph_feature_path, source_psk2index_path,
                                 target_feature_path, target_sex_adj_path, target_venue_adj_path, target_train_mask_path,
                                 target_graph_feature_path, target_psk2index_path, is_train=True)
    train_loader.load()
    #print(train_loader.features.shape)

    test_loader = PatientLoader(config, source_feature_path, source_sex_adj_path, source_venue_adj_path,
                                 source_test_mask_path, source_graph_feature_path, source_psk2index_path,
                                 target_feature_path, target_sex_adj_path, target_venue_adj_path, target_test_mask_path,
                                 target_graph_feature_path, target_psk2index_path, is_train=False)
    test_loader.load()

    #print(test_loader.features.shape)

    # add num_iter_per_epoch to config for trainer
    config = update_config_by_datasize(config, train_loader.get_datasize(),
                                       test_loader.get_datasize(),
                                       train_loader.get_feature_size())

    tfconfig = tf.ConfigProto()
    # specify GPU usage if using GPU
    #tfconfig.gpu_options.allow_growth = True
    #tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4

    # create tensorflow session
    with tf.Session(config=tfconfig) as sess:
        # create an instance of the model you want
        model = Model(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = Trainer(sess, model, train_loader, test_loader, config, logger)
        # load model if exists
        # model.load(sess)
        # here you train your model
        trainer.train()

    # tester


if __name__ == '__main__':
    main()
