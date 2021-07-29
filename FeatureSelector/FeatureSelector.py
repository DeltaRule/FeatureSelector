import numpy as np
import torch 
import torch.nn as nn
from typing import Union
from typing_extensions import Literal

class Mlp(nn.Module):
    def __init__(self, inSize:int, hiddenSize:int,outputSize:int,outputFuntion:nn = nn.Sigmoid,skaling:float = 10.0, layerDepth:int = 4, dropout:float = 0.0):
        """Create a Pytorch Model for the Wrapper part

        Args:
            inSize (int): the number of features
            hiddenSize (int): number of Hidden Size Neurons
            outputSize (int): output dimension
            outputFuntion (torch.nn): output function. Defaults to nn.Sigmoid
            skaling (float, optional): the skaling of the SNE, recomended is 0.25 to 10. Defaults to 10.
            layerDepth (int, optional): number of Hidden layers. Defaults to 4.
            dropout (float, optional): If dropout should be used if 0.0 dropout wont be used. Defaults to 0.0.
        """
        super(Mlp, self).__init__()
        sne = []
        sne.append(nn.Linear(inSize,int(inSize*skaling)))
        sne.append(nn.ReLU())
        if(dropout != 0.0):
            sne.append(nn.Dropout(dropout))
        sne.append(nn.Linear(int(inSize*skaling),int(inSize*skaling)))
        sne.append(nn.ReLU())
        if(dropout != 0.0):
            sne.append(nn.Dropout(dropout))
        sne.append(nn.Linear(int(inSize*skaling), inSize))
        sne.append(nn.Hardsigmoid())
        #this is where the magic happens, this is the SNE Block which will give weights to each feature
        self.sne = nn.Sequential(*sne)

        decisionModel = []
        #Input layer
        decisionModel.append(nn.Linear(inSize,hiddenSize))
        decisionModel.append(nn.ReLU())
        if(dropout != 0.0):
            decisionModel.append(nn.Dropout(dropout))
        #hidden layer 
        for _ in range(layerDepth):
            decisionModel.append(nn.Linear(hiddenSize,hiddenSize))
            decisionModel.append(nn.ReLU())
            if(dropout != 0.0):
                decisionModel.append(nn.Dropout(dropout))
        #outputLayer
        decisionModel.append(nn.Linear(hiddenSize, outputSize))
        decisionModel.append(outputFuntion())
        self.decisionModel = nn.Sequential(*decisionModel)

    def forward(self, x):
        w = self.sne(x)
        return self.decisionModel(w*x), w


class FeatureSelector():
    def __init__(self,  numberOfFeatures:int, 
                        toDelPerStep:int = 1,
                        iterations:int = 1,
                        hiddenSize: int = 70,
                        outputFunction: "nnLoss" = nn.Sigmoid,
                        skalingLayer: float = 10.0,
                        layerDepth: int = 4,
                        dropout: float = 0.0,
                        device: Literal["auto", "cuda", "cpu"] = "auto",
                        loss:Union["nnLoss", Literal["mse", "bce"]] = "mse",
                        optimizer:Literal["sgd", "rmsprop"] = "sgd", 
                        learnRate:float = 0.01,
                        momentum: float = 0.9,
                        weightDecay: float = 0.0,
                        batchSize : int = 100,
                        stopRange: int = 10,
                        verbose: Literal[0,1,2] = 1,
                        dataSkaling: Literal["auto", None, "minmax", "meanStd"] = "auto"):
        self.numberOfFeatures = numberOfFeatures
        self.toDelPerStep = toDelPerStep
        self.iterations = iterations
        self.hiddenSize = hiddenSize
        self.outputFunction = outputFunction
        self.skalingLayer = skalingLayer
        self.layerDepth = layerDepth
        self.dropout = dropout
        if(device == "auto"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(type(loss) == str):
            if(loss == "mse"):
                self.loss = nn.MSELoss()
            elif(loss == "bce"):
                self.loss = nn.BCELoss()
            else:
                self.loss = nn.MSELoss()
        else:
            self.loss = loss
        self.optimizer = optimizer
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.batchSize = batchSize
        self.stopRange = stopRange
        self.dataSkaling = dataSkaling
    def fit(self,X:np.ndarray,Y:np.ndarray,validationData:Union[Literal["auto"], tuple] = "auto"):
        if(validationData == "auto"):
            pass



        