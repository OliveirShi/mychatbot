"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
from seq2seq.config.config import BUCKETS

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    """Create vocabulary file (if it does not exist yet) from data file."""
    print("Creating vocabulary %s from data %s, vocabulary size: %d" % (vocabulary_path, data_path,
                                                                        max_vocabulary_size))
    vocab = {}
    with open(data_path, mode="r") as f:
        counter = 0
        for line in f:
            counter += 1
            tokens = line.strip().split()
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def read_vocabulary(vocab_path):
    """ read vocabulary from file
    :returns
        vocab: dict
        _vocab: list of reverse vocab
    """
    _vocab = []
    with open(vocab_path, 'r') as f:
        _vocab.extend(f.readlines())
    _vocab = [line.strip() for line in _vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(_vocab)])
    return vocab, _vocab


def sentence_to_token_ids(sentence, vocab):
    """ Convert a string to a list of integers representing token-ids"""
    words = sentence.strip().split()
    return [vocab.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path):
    """Tokenize data file and turn into token-ids using given vocabulary file."""
    _vocab = []
    with open(vocabulary_path, mode="r") as vocab_file:
        _vocab.extend(vocab_file.readlines())
    _vocab = [line.strip() for line in _vocab]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(_vocab)])
    with open(target_path, 'w') as target_file:
        with open(data_path, 'r') as data_file:
            for line in data_file:
                token_idxs = []
                for word in line.strip().split():
                    token_idxs.append(vocab_dict.get(word, UNK_ID))
                target_file.write(" ".join([str(tok) for tok in token_idxs]) + '\n')


def prepare_dialog_data(data_dir, vocabulary_size):
    """Get dialog data into data_dir, create vocabularies and tokenize data."""
    print(vocabulary_size)
    train_encode_file = data_dir + 'train.source'
    train_decode_file = data_dir + 'train.target'
    valid_encode_file = data_dir + 'valid.source'
    valid_decode_file = data_dir + 'valid.target'
    # test_encode_file = data_dir + 'test.source'
    # test_decode_file = data_dir + 'test.source'

    # Create vocabularies of the appropriate sizes.
    encode_vocab_path = data_dir + "encode.vocab"
    decode_vocab_path = data_dir + "decode.vocab"
    create_vocabulary(encode_vocab_path, train_encode_file, vocabulary_size)
    create_vocabulary(decode_vocab_path, train_decode_file, vocabulary_size)

    # Create token ids for the training data.
    train_idx_enc = data_dir + "train.idx.enc"
    train_idx_dec = data_dir + "train.idx.dec"
    data_to_token_ids(train_encode_file, train_idx_enc, encode_vocab_path)
    data_to_token_ids(train_decode_file, train_idx_dec, decode_vocab_path)

    # Create token ids for the development data.
    valid_idx_enc = data_dir + "valid.idx.enc"
    valid_idx_dec = data_dir + "valid.idx.dec"
    data_to_token_ids(valid_encode_file, valid_idx_enc, encode_vocab_path)
    data_to_token_ids(valid_decode_file, valid_idx_dec, decode_vocab_path)

    return ((train_idx_enc, train_idx_dec),
            (valid_idx_enc, valid_idx_dec),
            (encode_vocab_path, decode_vocab_path))


def read_data(source_path, target_path, max_size=None):
    """Read data from source file and put into buckets."""
    data_set = [[] for _ in BUCKETS]

    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_idx = [int(x) for x in source.split()]
                target_idx = [int(x) for x in target.split()]
                target_idx.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
                    if len(source_idx) < source_size and len(target_idx) < target_size:
                        data_set[bucket_id].append([source_idx, target_idx])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set