import torch
import torch.nn as nn
from torchtext import data
import numpy as np
import load_data
import torch.optim as optim
from cnn import Config, Model

n_folds = 4
fix_length = 50

POPULATION_SIZE = 10
DROPOUT_INIT = 0.5
DROPOUT_INTERVAL = 0.01
LEARNING_RATE = 0.001
EPOCH = 512
BATCH_SIZE = 256

# Choose which Dataset to load
# Dataset_name = "MR"
Dataset_name = "TREC"
#Dataset_name = "SST1"
#Dataset_name = "SST2"
train_test_generator, text_field = load_data.load_data(Dataset_name=Dataset_name, n_folds=n_folds, fix_length=fix_length)
pretrained_embeddings = text_field.vocab.vectors
train_set = []
test_set = []
for fold, (train_dataset, test_dataset) in enumerate(train_test_generator):

    # device: None for CPU, torch.device for GPU
    train_iter = data.Iterator(train_dataset, batch_size = len(train_dataset), sort=False, device=None, train=True, shuffle=True)
    test_iter = data.Iterator(test_dataset, batch_size = len(test_dataset), sort=False, device=None, train=False, shuffle=True)

    for batch in train_iter:
        train_set.append(batch)

    for batch in test_iter:
        test_set.append(batch)

(train_data, train_label), (test_data, test_label) = (train_set[0].text,train_set[0].label), (test_set[0].text, test_set[0].label)


def explore(h, n, interval):
    if h - interval * n/2 <= 0:
        h_lst = [interval * (i + 1) for i in range(n)]
    elif h + interval * n/2 >= 1:
        h_lst = [1 - interval * (i + 1) for i in range(n)]
    else:
        h_lst = [h + interval * (i - n/2) for i in range(n)]
    return h_lst


def evaluate1(net): 
    loss = nn.CrossEntropyLoss()
    input = net(train_data)
    return -loss(input, train_label).item()


def evaluate2(net): 
    return accuracy(net, train_data, train_label)

def accuracy(net, data, label):
    result = np.argmax(net(data).detach().numpy(), axis=1)
    correct_count = 0
    for i in range(len(data)):
        if result[i] == label[i]:
            correct_count += 1
    return correct_count/len(data)


def step(train_data, train_label, batch_size, net, optimizer):
    for i in range(0, len(train_data), batch_size):
        if i + batch_size > len(train_data):
            batch_data = train_data[i:]
            batch_label = train_label[i:]
        else:
            batch_data = train_data[i:i+batch_size]
            batch_label = train_label[i:i+batch_size]
        pred = net(batch_data)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(pred, batch_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


training_accs = open('training accuracy.txt', 'w')
test_accs = open('test accuracy.txt', 'w')
loss_lst = open('loss.txt', 'w')
dropout_lst = open('dropout.txt', 'w')
training_accs.close()
test_accs.close()
loss_lst.close()
dropout_lst.close()

for i in range(n_folds):
    (train_data, train_label), (test_data, test_label) = (train_set[i].text,train_set[i].label), (test_set[i].text, test_set[i].label)
    train_data = train_data.T
    test_data = test_data.T

    config = Config(pretrained_embeddings, DROPOUT_INIT)
    nets = [Model(config) for _ in range(POPULATION_SIZE)]
    population = [(net, evaluate1(net), DROPOUT_INIT) for net in nets]

    for e in range(EPOCH):
        population.sort(key=lambda p: p[1], reverse=True)
        elite = population[0]
        elite_net = elite[0]

        training_acc = accuracy(elite_net, train_data, train_label)
        test_acc = accuracy(elite_net, test_data, test_label)
        print("Epoch {}, training accuracy {}, test accuracy {}".format(e, training_acc, test_acc))
        training_accs = open('training accuracy.txt', 'a')
        test_accs = open('test accuracy.txt', 'a')
        loss_lst = open('loss.txt', 'a')
        dropout_lst = open('dropout.txt', 'a')
        training_accs.write('%s\n' % training_acc)
        test_accs.write('%s\n' % test_acc)
        loss_lst.write('%s\n' % -evaluate1(elite_net))
        dropout_lst.write('%s\n' % elite[2])
        training_accs.close()
        test_accs.close()
        loss_lst.close()
        dropout_lst.close()
        
        dropout_lst = explore(elite[2], POPULATION_SIZE, DROPOUT_INTERVAL)

        population = []
        for j in range(POPULATION_SIZE):
            config = Config(pretrained_embeddings, dropout_lst[j])
            net = Model(config)
            net.load_state_dict(elite_net.state_dict())
            optimizer = optim.Adadelta(net.parameters(), lr=LEARNING_RATE)
            step(train_data, train_label, BATCH_SIZE, net, optimizer)
            if e > EPOCH/2:
                population.append((net, evaluate2(net), dropout_lst[j]))
            else:
                population.append((net, evaluate1(net), dropout_lst[j]))
