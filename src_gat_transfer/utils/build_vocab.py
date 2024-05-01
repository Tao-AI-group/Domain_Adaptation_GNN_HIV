"""
build_vocab
"""
from pathlib import Path
from collections import Counter
import string
import sys
from utils.utils import get_args, save_dict_to_json
from utils.config import get_config_from_json, update_config_by_vocab

PAD_WORD = '</s>'
PAD_TAG = 'O'

def build_vocab(X, MINCOUNT=1):
    counter_words = Counter()
    for words in X:
        # print(words)
        new_words = []
        for word in words:
            # convert a bytes like object to string, and remove punctuations
            # t = str(word,sys.stdout.encoding).translate(str.maketrans("", "", string.punctuation))
            t = str(word,sys.stdout.encoding)
            if t != '':
                new_words.append(t)
        counter_words.update(new_words)
    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}
    return vocab_words

def load_vocab(path, mode='word'):
    vocab = {}
    index = 0
    with Path(path).open('r') as f:
        for l in f:
            key = l.strip() if mode == 'word' else int(l.strip())
            vocab[key] = index
            index += 1
    return vocab


if __name__ == '__main__':
    """
    load data and build vocab
    """
    # load data from disk
    config_path = '../configs/text_ner.json'
    config = get_config_from_json(config_path)
    datadir = '../data_samples/ner/'
    words_path = datadir + 'corpus_words.txt'
    tags_path = datadir + 'corpus_tags.txt'
    X, Y = [], []
    with Path(words_path).open('rb') as f:
        for l in f:
            X.append(l.strip().split())
    word_vocab = build_vocab(X)
    with Path(tags_path).open('rb') as f:
        for l in f:
            Y.append(l.strip().split())
    tag_vocab = build_vocab(Y)
    # add padding token
    if PAD_WORD not in word_vocab: word_vocab.add(PAD_WORD)
    if PAD_TAG not in tag_vocab: tag_vocab.add(PAD_TAG)
    # save to disk
    word_vocab_path = datadir + 'word_vocab.txt'
    tag_vocab_path = datadir + 'tag_vocab.txt'
    with Path(word_vocab_path).open('w') as f:
        f.write('\n'.join(word for word in word_vocab))
    with Path(tag_vocab_path).open('w') as f:
        f.write('\n'.join(tag for tag in tag_vocab))
    # save json config
    word_vocab_size = len(word_vocab)
    tag_vocab_size = len(tag_vocab)
    config = update_config_by_vocab(config, word_vocab_size, tag_vocab_size)
    save_dict_to_json(config, config_path)
    print("updated config file by updating vocabulary")

