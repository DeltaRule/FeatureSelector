# Feature Selector

This is a Feature Selector for 1 Dimensional Data.

## Description

This Feature Selector uses a Pytorch Model with a 1D SNE Block wich are the attention weights used for the backward wrapper to eliminate bad Features. This Method isn't that fast, so please consider using the pip package feature-selector first to get there the best 20 Features and then using my Package to get the best features out of that

## Getting Started

### Dependencies

- Numpy and Pytorch

### Installing

- You can clone the github, and I will probably put a pip link aswell

### Executing program

- How to run the program

```python
from FeatureSelector import FeatureSelector
ft = FeatureSelector(10)# the number of features to reduce to
print(ft.fit(X,Y))
```

- For initilizing the Clas you have these Parameters to choose from

**numberOfFeatures (int)**: the number of features it should reduce to per iteration. Meaning at the end it can come out with more features then this number here, depending on the number of iterations

**toDelPerStep (int, optional)**: This is still a backward wrapper method of removing the features. This means How many Features should be removed for each learning of a Neural Network. A Higher Number means the programm will be faster, but maybe not to accurate depending on the data . Defaults to 1.

**iterations (int, optional)**: How often schould the Programm run from the beginning. The more often it runs the more stable the output will be, but it will take longer. Defaults to 2.

**hiddenSize (int, optional)**: the Hidden Size of the Klassifikation Part of the Model. Defaults to 71.

**outputFunction (nnLoss, optional)**: the Output Funktion of the Model, if you don't want to have an Outputfunktion put nn.Identity(). Defaults to nn.Sigmoid.

**skalingLayer (float, optional)**: the Skaling Factor of the Attention part of the model (hiddensize of attention = skaling\* insize), probably should be 1.0-10.0. Defaults to 10.0.

**layerDepth (int, optional):** the number of Hiddenlayers in the Classifiaction part of the model. Defaults to 4.

**dropout (float, optional):** the dropout of the model, Out of experience this doesn't matter to much for my examples. Defaults to 0.0.

**device (Literal["auto", "cuda", "cpu"], optional)**: the device the model should be trained on, if "auto", it will look if you have a cuda system or not and decide on there own. other Values you can choose from are "cpu", "cuda". Defaults to "auto".

**loss (Union["function", Literal["mse", "bce"]], optional)**: the loss function that it should be trained on. you can just put your own funtin in there, in the form of pytorch Loss funtions like torch.nn.F.mse_loss. Defaults to "mse".

**validationMetric (function, optional)**: Its recomended to use Something like BER, buts thats for the Specific Tasks, But it won't work if you use a Metric like Accuracy, because the code needs a Metric where the lower the metric the better. Defaults to F.mse_loss.

**optimizer (Literal["optimizerClass","sgd", "rmsprop"], optional)**: which optimizer to use for this, please use an optimizer that has momentum and weight decay from torch.optim, else that will raise an error probablys. Defaults to "sgd".

**learnRate (float, optional)**: the learnrate for the optimizer. Defaults to 1.01.

**momentum (float, optional)**: the momentum for the optimizer(if you want to use RMSProp use a lower momentum). Defaults to 1.9.

**weightDecay (float, optional)**: the weight decay for the optimizer. Defaults to 0.0.

**batchSize (int, optional)**: the batch size to train and evalute the mode. Defaults to 100.

**maxEpochs (int, optional)**: the maximal epochs to train the model. Defaults to 800.

**patience (int, optional)**: This is for early Stopping the training, if the number is 6 for example, then it will test if the smoothed validation metric curve raises the value for 6 successively values. Defaults to 6.

**verbose (Literal[0,1,2], optional)**: the amount of information that is printed out, the higher the number the more information is printed out but choose between 0,1,2. Defaults to 1.

**dataSkaling (Literal["auto", None, "minMax", "meanStd"], optional)**: the skaling of the input data, "auto" just meaning there will be a min max Skale, if you want to use a custom skaling then skale the data beforehand and put None as a skaling. Defaults to "auto".

## Authors

Contributors names and contact info
ex. Ilja Dontsov
ex. [@deltaRule](https://www.delta-rule.com)

## Version History

- 0.1
- Initial Release
- See [commit change]() or See [release history]()
- 0.1
- Initial Release
