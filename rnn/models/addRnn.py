from __future__ import print_function
import numpy as np
from time import sleep
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BinaryAddRNN (nn.Module):
    def __init__(self, hidd_dim, layer_num=1):
        super(BinaryAddRNN, self).__init__()

        self.inputDim=2

        self.hidd_dim=hidd_dim

        self.outputDim = 1

        self.rnnLayer = nn.RNN(2, hidd_dim, layer_num)

        self.fcLayer = nn.Linear(hidd_dim, 1)

        self.activate=nn.Sigmoid()

    def forward(self, x ):
        
        #print("x", x, x.size())
        rnnLayeOutr,_ =self.rnnLayer(x) 
        #print("rnnLayeOutr", rnnLayeOutr, rnnLayeOutr.size())
        #T,B,D  = lstmOut.size(0),lstmOut.size(1) , lstmOut.size(2)
        rnnLayeOutr = rnnLayeOutr.contiguous() 
        
        #lstmOut = lstmOut.view(B*T, D)
        output=self.fcLayer(rnnLayeOutr)
        #print("output", output, output.size())
        #outputLayerActivations=outputLayerActivations.view(T,B,-1).squeeze(1)

        output = self.activate(output)
        #print("output", output)
        
        
        #assert 1==2

        return output

def BinaryAddRNN8_1():
    return BinaryAddRNN(8, 1)

def BinaryAddRNN8_2():
    return BinaryAddRNN(8, 2)

    

def loadFileRawData(datapath = "data.txt", PERMUTAION = False):
    
    filename = datapath
    a_list = []
    b_list = []
    c_list = []
    def str_2_list(data_list, orderList):
        ret_list = []
        for i in orderList:
            tmp_list = data_list[i].strip().split(" ")
            tmp_ret_list = [int(tmp_list[0][1]),int(tmp_list[1]),int(tmp_list[2]),int(tmp_list[3]),int(tmp_list[4]),int(tmp_list[5]),int(tmp_list[6]),int(tmp_list[7][0])]
            ret_list.append(tmp_ret_list)
        return ret_list
    with open(filename, "r") as file:
        filein = file.read().splitlines()
        for item in filein:
            tmp_list = item.strip().split(",")
            a_list.append(tmp_list[0])
            b_list.append(tmp_list[1])
            c_list.append(tmp_list[2])
    if PERMUTAION:
        order = np.random.permutation(len(a_list))
    else:
        order = list(range(len(a_list)))
    a_list = str_2_list(a_list, order)
    b_list = str_2_list(b_list, order)
    c_list = str_2_list(c_list, order)
    return a_list, b_list, c_list

def loadOneSample(a , b, c, bugSample = False):
        # return tensor(8,2) (8,1)
        if bugSample:
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            c = np.asarray(c, dtype=np.float32).reshape((8,1))
            return np.c_[a,b].reshape(8,2), c

        a = np.asarray(a[::-1], dtype=np.float32).reshape(8,1)
        b = np.asarray(b[::-1], dtype=np.float32).reshape(8,1)
        c = np.asarray(c[::-1], dtype=np.float32).reshape((8,1))
        return np.c_[a,b], c

def trainOneModel(Bug = False, hidden_num = 8, layer_num = 1, epoch_num=4):
    rawData = loadFileRawData()
    model = BinaryAddRNN(hidden_num ,layer_num)
    lossFunction = nn.BCELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    totalLoss = 0
    epoches = epoch_num
    for epoch in range(epoches):
        for i in range(4000):
            x, y = loadOneSample(rawData[0][i],rawData[1][i],rawData[2][i], Bug)
            
            model.zero_grad()

            x_var=autograd.Variable(torch.from_numpy(x).unsqueeze(1).float()) #convert to torch tensor and variable
            
            x_var= x_var.contiguous()
            
            y_var=autograd.Variable(torch.from_numpy(y))
            
            finalScores = model(x_var)
            
            loss=lossFunction(finalScores,y_var)  
            
            totalLoss+=loss.item()
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

        print(loss.item())

    testResult = 0
    for i in range(4000, 5000):
        x, y = loadOneSample(rawData[0][i],rawData[1][i],rawData[2][i], Bug)
        
        model.zero_grad()

        x_var=autograd.Variable(torch.from_numpy(x).unsqueeze(1).float()) #convert to torch tensor and variable
        
        x_var= x_var.contiguous()
        
        y_var=autograd.Variable(torch.from_numpy(y))

        finalScores = model(x_var)
    
        finalScores=finalScores.gt(0.5).view(8)

        y_var = y_var.view(8).byte()
        
        print()
        print("##############")
        print("{} = {} + {}\n{}, {}".format(finalScores, rawData[0][i][::-1], rawData[1][i][::-1], y_var, rawData[2][i][::-1]))
        if torch.equal(finalScores, y_var):
            testResult += 1
    
    print("Accuracy, {}".format(testResult/1000))

    return model, testResult/1000
    



def saveModel(model ,PATH):
    torch.save(model.state_dict(), PATH)

def modelFilePathAndName(modelname, layer_num, hidden_num, accuracy, timestamp):
    path = "{}_{}_{}".format(modelname,layer_num,hidden_num)
    # addrnn_good_8_1
    fileName = "{}_{}_model.t7".format(timestamp, accuracy)

    return path, fileName

if __name__ == "__main__":
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    net, testAcc = trainOneModel(hidden_num = 8, layer_num = 1, epoch_num=4)
    pathPrefix, fileName = modelFilePathAndName("addrnn_good", 8, 1, testAcc, timestamp)
    path = "../trained_nets/{}".format(pathPrefix)
    print("model will save to {}".format(path))
    import os
    if not os.path.exists(path):
      os.makedirs(path)
    saveModel(net, path + "/" + fileName)