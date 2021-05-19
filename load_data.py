import torch
import torch.nn as nn
import torchwordemb
import numpy as np
import pandas
import re
import os
import random
import tarfile
import urllib
from torchtext import data
from collections import defaultdict
import spacy
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors

spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def download(url, filename, dirname):
    dirpath = os.path.join('.', dirname)
    if not os.path.isdir(dirpath):
        filepath = os.path.join('.', filename)

        # Dataset not exist, download from url
        if not os.path.isfile(filepath):
            print("Start Downloading ...")
            urllib.request.urlretrieve(url, filepath)

        # Has .tar file, start to extract
        with tarfile.open(filepath, 'r') as f:
            print("Start Extracting ...")
            f.extractall('.')

    return os.path.join(dirpath, '')

def string_clean(string):

    # String cleaning, lower cased
    string = re.sub(r"[^A-Za-z0-9,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

# Load directly from website (download and load)
def load_MR(fields, dirname, fix_length):

    vocab = defaultdict(float)

    with open(os.path.join(dirname, 'rt-polarity.neg'), errors='ignore') as f:
        examples = []
        for line in f:
            examples += [data.Example.fromlist([line, 'negative'], fields)]
            words = set(string_clean(line).split())
            for word in words:
                vocab[word] += 1

    with open(os.path.join(dirname, 'rt-polarity.pos'), errors='ignore') as f:
        for line in f:
            examples += [data.Example.fromlist([line, 'positive'], fields)]
            words = set(string_clean(line).split())
            for word in words:
                vocab[word] += 1

    random.shuffle(examples)

    return examples

def load_from_pkl(fields, pkl_name):

    train_examples = []
    test_examples  = []
    # Read from pickles in list form
    corpus = pandas.read_pickle(pkl_name)
    splits, sentences, labels = list(corpus.split), list(corpus.sentence), list(corpus.label)

    for split, line, label in zip(splits, sentences, labels):
        if split == 'train'or split == 'dev':
            train_examples += [data.Example.fromlist([line, label], fields)]
        elif split == 'test':
            test_examples += [data.Example.fromlist([line, label], fields)]

        else:
            print("New Type of split: " + split )
            return -1
    random.shuffle(train_examples)
    random.shuffle(test_examples)

    return train_examples, test_examples

def postprocessing_label(arr, vocab):
    return [(i-1) for i in arr]

def load_data(Dataset_name, n_folds, fix_length):

    text_field = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=fix_length)
    label_field = data.Field(sequential=False, postprocessing=postprocessing_label)
    text_field.preprocessing = data.Pipeline(string_clean)
    fields= [('text', text_field), ('label', label_field)]

    if Dataset_name == "MR":
        pickle_name = "./Data/MR.pkl"

    elif Dataset_name == "TREC":
        pickle_name = "./Data/TREC.pkl"

    elif Dataset_name == "SST1":
        pickle_name = "./Data/SST1.pkl"

    elif Dataset_name == "SST2":
        pickle_name = "./Data/SST2.pkl"

    else:
        print(Dataset_name + " Not Yet Implement!")
        return -1

    train_examples, test_examples = load_from_pkl(fields=fields, pkl_name=pickle_name)

    print("Finish Loading " + Dataset_name + " Dataset")
    print("Length of training data: " + str(len(train_examples)))
    print("Length of testing data: " + str(len(test_examples)))
    train_data = data.Dataset(train_examples, fields)
    test_data  = data.Dataset(test_examples , fields)

    # Dictionary from golve.6B.100d
    #text_field.build_vocab(train_data, vectors="glove.6B.100d")

    # Dictionary from google news negative 300
    bin_filepath = "GoogleNews-vectors-negative300.bin"
    w2v, vec = torchwordemb.load_word2vec_bin(bin_filepath)

    text_field.build_vocab(train_data, test_data)
    text_field.vocab.set_vectors(stoi=w2v, vectors=vec, dim=300)

    label_field.build_vocab(train_data, test_data)

    ########## Return Iterator Method ##########
    '''
    # device: -1 for CPU, 0 for GPU
    #train_iter, test_iter = data.Iterator.splits((train_data, test_data), sort=False, batch_size=1, device = -1)

    for batch in train_iter:
        print("Batch text: ", batch.text)
        print("Batch label: ", batch.label)

    for batch in test_iter:
        print("Batch text: ", batch.text)
        print("Batch label: ", batch.label)

    return train_iter, test_iter
    '''
    if n_folds == 1:
        def single_dataset():
            train_exs_arr = np.array(train_examples)
            test_exs_arr = np.array(test_examples)
            yield(
                    data.Dataset(train_exs_arr, fields),
                    data.Dataset(test_exs_arr , fields)
            )
        return single_dataset(), text_field

    else:
        ########## Return Iter Folds Method ##########
        kf = KFold(n_splits=n_folds)

        def iter_folds():
            train_exs_arr = np.array(train_examples)
            for train_idx, val_idx in kf.split(train_exs_arr):
                yield (
                        data.Dataset(train_exs_arr[train_idx], fields),
                        data.Dataset(train_exs_arr[val_idx], fields),
                )
        return iter_folds(), text_field
