import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def update_config_by_summary(config):
    experiments_dir = config.exp_dir
    config.summary_dir = os.path.join(experiments_dir, config.exp_name, "summary/")
    config.model_dir = os.path.join(experiments_dir, config.exp_name, "model/")
    return config


def update_config_by_vocab(config, word_vocab_size, tag_vocab_size):
    config.word_vocab_size = word_vocab_size
    config.tag_vocab_size = tag_vocab_size
    return config


def update_config_by_datasize(config, train_size, test_size, feature_size):
    # config.num_iter_per_epoch = int(train_size / config.batch_size)
    config.train_size = train_size
    config.test_size = test_size
    config.feature_size = feature_size
    return config

