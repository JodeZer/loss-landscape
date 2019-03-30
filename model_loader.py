import os
import cifar10.model_loader
import rnn.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == "binaryAdd":
        net = rnn.model_loader.load(model_name, model_file, data_parallel)
    return net
