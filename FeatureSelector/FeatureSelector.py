import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from typing_extensions import Literal
import warnings
from tqdm import tqdm
import os 

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
                        loss:Union["function", Literal["mse", "bce"]] = "mse",
                        validationMetric:"function" = F.mse_loss,
                        optimizer:Literal["optimizerClass","sgd", "rmsprop"] = "sgd", 
                        learnRate:float = 0.01,
                        momentum: float = 0.9,
                        weightDecay: float = 0.0,
                        batchSize : int = 100,
                        maxEpochs : int = 800,
                        patience: int = 6,
                        verbose: Literal[0,1,2] = 1,
                        dataSkaling: Literal["auto", None, "minMax", "meanStd"] = "auto"):
        """Initialize the main class

        Args:
            numberOfFeatures (int): the number of features it should reduce to per iteration. Meaning at the end it can come out with more features then this number here, depending on the number of iterations
            toDelPerStep (int, optional): This is still a backward wrapper method of removing the features. This means How many Features should be removed for each learning of a Neural Network. A Higher Number means the programm will be faster, but maybe not to accurate depending on the data . Defaults to 1.
            iterations (int, optional): How often schould the Programm run from the beginning. The more often it runs the more stable the output will be, but it will take longer. Defaults to 1.
            hiddenSize (int, optional): the Hidden Size of the Klassifikation Part of the Model. Defaults to 70.
            outputFunction (nnLoss, optional): the Output Funktion of the Model, if you don't want to have an Outputfunktion put nn.Identity(). Defaults to nn.Sigmoid.
            skalingLayer (float, optional): the Skaling Factor of the Attention part of the model (hiddensize of attention = skaling* insize), probably should be 1.0-10.0. Defaults to 10.0.
            layerDepth (int, optional): the number of Hiddenlayers in the Classifiaction part of the model. Defaults to 4.
            dropout (float, optional):  the dropout of the model, Out of experience this doesn't matter to much for my examples. Defaults to 0.0.
            device (Literal["auto", "cuda", "cpu"], optional): the device the model should be trained on, if "auto", it will look if you have a cuda system or not and decide on there own. other Values you can choose from are "cpu", "cuda". Defaults to "auto".
            loss (Union["function", Literal["mse", "bce"]], optional): the loss function that it should be trained on. you can just put your own funtin in there, in the form of pytorch Loss funtions like torch.nn.F.mse_loss. Defaults to "mse".
            validationMetric (function, optional): Its recomended to use Something like BER, buts thats for the Specific Tasks, But it won't work if you use a Metric like Accuracy, because the code needs a Metric where the lower the metric the better. Defaults to F.mse_loss.
            optimizer (Literal["optimizerClass","sgd", "rmsprop"], optional): which optimizer to use for this, please use an optimizer that has momentum and weight decay from torch.optim, else that will raise an error probablys. Defaults to "sgd".
            learnRate (float, optional): the learnrate for the optimizer. Defaults to 0.01.
            momentum (float, optional): the momentum for the optimizer(if you want to use RMSProp use a lower momentum). Defaults to 0.9.
            weightDecay (float, optional): the weight decay for the optimizer. Defaults to 0.0.
            batchSize (int, optional): the batch size to train and evalute the mode. Defaults to 100.
            maxEpochs (int, optional): the maximal epochs to train the model. Defaults to 800.
            patience (int, optional): This is for early Stopping the training, if the number is 6 for example, then it will test if the smoothed validation metric curve raises the value for 6 successively values. Defaults to 6.
            verbose (Literal[0,1,2], optional): the amount of information that is printed out, the higher the number the more information is printed out but choose between 0,1,2. Defaults to 1.
            dataSkaling (Literal["auto", None, "minMax", "meanStd"], optional): the skaling of the input data, "auto" just meaning there will be a min max Skale, if you want to use a custom skaling then skale the data beforehand and put None as a skaling. Defaults to "auto".
        """
        self.numberOfFeatures = numberOfFeatures
        self.toDelPerStep = toDelPerStep
        self.iterations = iterations
        self.hiddenSize = hiddenSize
        self.outputFunction = outputFunction
        self.skalingLayer = skalingLayer
        self.layerDepth = layerDepth
        self.dropout = dropout
        if(device == "auto"):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if(type(loss) == str):
            if(loss == "mse"):
                self.loss = F.mse_loss
            elif(loss == "bce"):
                self.loss = F.binary_cross_entropy
            else:
                warnings.warn(f"Couldn't interpret {loss}, going to go with the default mse")
                self.loss = F.mse_loss
        else:
            self.loss = loss
        self.validationMetric = validationMetric
        if(type(optimizer) == str):
            if(optimizer == "sgd"):
                self.optimizer = torch.optim.SGD
            elif(optimizer == "rmsprop"):
                self.optimizer = torch.optim.RMSprop
            else:
                warnings.warn(f"Couldn't interpret {optimizer}, going to go with the default sgd")
                self.optimizer = torch.optim.SGD
        else:
            self.optimizer = optimizer
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.batchSize = batchSize
        self.maxEpochs = maxEpochs
        self.patience = patience
        self.verbose = verbose
        self.dataSkaling = dataSkaling
        if(self.dataSkaling not in ["auto", None, "minMax", "meanStd"]):
            warnings.warn(f"Couldn't interpret {self.dataSkaling}, going to go with no normailzation")
        #this is for the early Stopping method, this will smoothe out the curve, so we can see if the model is overfitting or not
        #more information @ https://en.wikipedia.org/wiki/Savitzky–Golay_filter
        self.savitzkyGolayFilter:np.ndarray = np.array([15, -55, 30, 135, 179, 135, 30, -55, 15])/429
        self.bestDict = {}
        
    def fit(self,X:np.ndarray,Y:np.ndarray,validationData:Union[Literal["auto"], tuple] = "auto")->dict:
        """The Funktion Called to do the reduction of data

        Args:
            X (np.ndarray): the input data 
            Y (np.ndarray): the target data
            validationData (Union[Literal["auto"], tuple] optional): if auto it will do a random 20% split of the input data, but you can also fill in ur custom data as a tuple meaning (valX,ValY). Defaults to "auto".

        Returns:
            dict: the best indicies and how often they came up per iteration (the more often a feature is recognised as a better feature the better it probobly is)
        """        
        #good practice
        self.randomSeed = np.random.randint(0,1000)
        np.random.seed(self.randomSeed)
        if(self.verbose > 1):
            print(f"random seed for this run is {self.randomSeed}")
        #If auto Split these two with a 20% split ratio, else use the data provided
        if(validationData == "auto"):
            testTrainSplit = np.random.choice([True, False], X.shape[0], p = [0.8,0.2])
            self.trainIn = X[testTrainSplit == True]
            self.trainOut = Y [testTrainSplit == True]
            self.valIn = X[testTrainSplit == False]
            self.valOut = Y[testTrainSplit == False]
            if(self.verbose > 1):
                print(f"The test train split produced these shapes train: {self.trainIn.shape}, validate: {self.valIn.shape}")
        else:
            self.trainIn = X 
            self.trainOut = Y  
            self.valIn, self.valOut = validationData

        if(self.verbose > 1):
            print(f"normalizing data")
        #normalize the data so that the Network can work with it(Neural Networks don't work good with non Normalized Data)
        self.normalize_data()
        bestData = {}
        if(self.verbose> 0 ):
            #remove the Data the iterations are just to remove randomness form the data 
            
            for iterNumber in range(self.iterations):
                print(f"Iteration: {iterNumber}")
                self.chooseList = [i for i in range(self.trainIn.shape[1])]
                for _ in tqdm(range(0, self.trainIn.shape[1] - self.numberOfFeatures, self.toDelPerStep)):
                    rankingDict = self.remove_feature()
                self.randomSeed = np.random.randint(0,1000)
                np.random.seed(self.randomSeed)
                if(self.verbose > 1):
                    print(f"random seed for next iteration is {self.randomSeed}")
                print(f"it {iterNumber} best Indicies: {self.chooseList}")
                for bestIdx in rankingDict:
                    if(rankingDict[bestIdx] != 0):
                        if(bestIdx in bestData):
                            bestData[bestIdx] += rankingDict[bestIdx]
                        else:
                            bestData[bestIdx] = rankingDict[bestIdx]
                
        else:
            for _ in range(self.iterations):
                self.chooseList = [i for i in range(self.trainIn.shape[1])]
                for _ in range(0, self.trainIn.shape[1] - self.numberOfFeatures, self.toDelPerStep):
                    rankingDict = self.remove_feature()
                self.randomSeed = np.random.randint(0,1000)
                np.random.seed(self.randomSeed)
                for bestIdx in rankingDict:
                    if(rankingDict[bestIdx] != 0):
                        if(bestIdx in bestData):
                            bestData[bestIdx] += rankingDict[bestIdx]
                        else:
                            bestData[bestIdx] = rankingDict[bestIdx]

        if(self.verbose > 0 ):
            print(f"the best indicies are: {bestData}")
        return bestData


    def normalize_data(self)->None:
        """normalize the data if needed so it get better results
        """        
        if(self.dataSkaling.lower() == "auto" or self.dataSkaling.lower() == "minmax"):
            maximum = np.amax(np.concatenate([self.trainIn,self.valIn]), axis = 0)
            minimum = np.amin(np.concatenate([self.trainIn,self.valIn]), axis = 0)
            self.trainIn = (self.trainIn - minimum)/(maximum - minimum)
            self.valIn = (self.valIn - minimum)/(maximum - minimum)
        elif(self.dataSkaling.lower() == "meanstd"):
            mean = np.mean(np.concatenate([self.trainIn,self.valIn]), axis = 0)
            std = np.std(np.concatenate([self.trainIn,self.valIn]), axis = 0)
            self.trainIn = (self.trainIn - mean)/(std)
            self.valIn = (self.valIn - mean)/(std)
    
    def remove_feature(self)->dict:
        """Learn the network and then remove the worst feature based on the attention weights
        """        
        
        #which Data is already not in the Data
        toDel = list(set([ i for i in range(self.trainIn.shape[1])]) -  set(self.chooseList))
        #Make Pytorch Tensors
        trainIn = torch.from_numpy(np.delete(self.trainIn, toDel, 1))
        trainOut = torch.from_numpy(self.trainOut)
        valIn = torch.from_numpy(np.delete(self.valIn, toDel, 1))
        valOut = torch.from_numpy(self.valOut)
        #set Seed so the Network is the same Initialized
        torch.manual_seed(self.randomSeed)
        Net = Mlp(valIn.shape[1],hiddenSize=self.hiddenSize, outputSize=self.trainOut.shape[1],outputFuntion=self.outputFunction, skaling=self.skalingLayer, layerDepth=self.layerDepth, dropout=self.dropout)
        Net.double()
        Net = Net.to(device=self.device)
        opt = self.optimizer(Net.parameters(), lr= self.learnRate, momentum=self.momentum, weight_decay=self.weightDecay)
        #these are to deternime an early stopping in case of overfitting, the smoothe filter is to filter the loss function output, and the get Gradient is to look if the smoothed Data has a uphill gradient, if so then I will early Stop
        getGradient = np.zeros(self.patience) + 1
        smootheFilter = np.zeros(9) + 1
        minError = 1
        trainDataSet = torch.utils.data.TensorDataset(trainIn, trainOut)
        trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=self.batchSize, shuffle=True, drop_last=True)
        valDataSet = torch.utils.data.TensorDataset(valIn, valOut)
        valLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=self.batchSize, shuffle=False, drop_last=False)
        
        for epoch in range(self.maxEpochs):
            #train 1 epoch
            for trainInBatch, trainOutBatch in trainLoader:
                opt.zero_grad()
                
                trainInBatch = trainInBatch.to(device=self.device)
                trainOutBatch = trainOutBatch.to(device=self.device)
                # print(self.device,trainInBatch.device )

                out, _ =  Net(trainInBatch)
                l = self.loss(out,trainOutBatch )
                l.backward()
                opt.step() 
            Net.eval()
            #test the Data on Validation Dataset
            valMetricSum = 0
            with torch.no_grad():
                for valInBatch, valOutBatch in valLoader:
                    valInBatch = valInBatch.to(device=self.device)
                    valOutBatch = valOutBatch.to(device=self.device)
                    out, _ = Net(valInBatch)
                    valMetricSum += self.validationMetric(out,valOutBatch ).item()
            #Smoothe the Data with the filter (.__init__)
            valMetricSum = valMetricSum/len(valLoader)
            smootheFilter = np.roll(smootheFilter, -1)
            smootheFilter[-1] = valMetricSum
            smoothedMetric = np.sum(smootheFilter * self.savitzkyGolayFilter)
            #Put it in the Gradient step so we can check for early stopping
            getGradient = np.roll(getGradient, -1)
            getGradient[-1] = smoothedMetric

            if(smoothedMetric < minError):
                torch.save(Net, "tmp.net")
                minError = smoothedMetric
            for i in range(len(getGradient)-1):
                if(getGradient[i+1] <= getGradient[i]):
                    break 
            else:
                break
            Net.train()
        #load best model
        Net = torch.load("tmp.net")
        Net.cpu()
        Net.eval()
        #remove tmp file
        os.remove("tmp.net")
        attW = []
        #now look at the attention of the SNE Block
        #(https://arxiv.org/abs/1709.01507)
        with torch.no_grad():


            _, W = Net(valIn)
            attW = W.detach().numpy()
        
        attW = np.mean(attW, axis=0)
        
        sortDict = {}
        sub = 0
        for i in range(self.trainIn.shape[1]):
            if(i in toDel):
                sortDict[i] = 0
                sub += 1
                continue

            sortDict[i] = attW[i-sub]
        sortDict = {k: v for k, v in sorted(sortDict.items(), key=lambda item: item[1])}
        cnt = 0
        for i in sortDict:
            if(sortDict[i] == 0):
                continue
            if(self.verbose > 1):
                print(f"removed index {i} @ epoch {epoch} with a value of {sortDict[i]}")
            self.chooseList.remove(i)
            cnt += 1
            if(cnt == self.toDelPerStep):
                break 
        return sortDict