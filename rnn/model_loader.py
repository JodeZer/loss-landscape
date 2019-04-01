import os
import torch, torchvision
import rnn.models.addRnn as rnn


# map between model name and function
models = {
    'binaryAddRnn8_1'                  : rnn.BinaryAddRNN8_1,
    'binaryAddRnn8_2'                  : rnn.BinaryAddRNN8_2,
    'binaryAddRnn16_2'                 : rnn.BinaryAddRNN16_2,
    'binaryAddRnn16_3'                 : rnn.BinaryAddRNN16_3,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
