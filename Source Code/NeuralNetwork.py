import numpy as np
import math as mt
from activationFunctions import *
from miscFunctions import doNothing, accuracyScore, defaultingFunc, MSE, R2, etaDefinerDefiner
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

def benchmark(X, z, parameters, NeuralNetwork=None, mode=None, randomSearch=None, writingPermissions=None, N=None):
    """Benchmarks Neural Networks using all permutations of given parameters
    and finds the optimal values. Functions for either classification or 
    regression, will write all iterations to file if given writing permissions.

    Args:
        X (numpy.ndarray): Input values
        z (numpy.ndarray): True values for inputs
        parameters (dict): dictionary of all parameters
        NeuralNetwork (NN, optional): Neural Network class to be benchmarked, Defaults to neural network in main script named NN.
        mode (string, optional): Determines whether network is benchmarked for "classification" or "regression". Defaults to "classification".
        writingPermissions (bool, optional): Determines whether script has permission to write to file. Defaults to False.

    Returns:
        (float, list, string): A tuple of best score, a list of parameters 
                               used, and a string containing those parameters
                               that can be printed
    """
    writingPermissions  = defaultingFunc(None, writingPermissions, False)
    mode                = defaultingFunc(None, mode, "classification")
    NeuralNetwork       = defaultingFunc(None, NeuralNetwork, NN)
    randomSearch        = defaultingFunc(None, randomSearch, False)
    N                   = defaultingFunc(None, N, 10000)



    dataSelection = np.random.choice(range(X.shape[0]), int(parameters["datafraction"]*X.shape[0]), replace=False)
    X = X[dataSelection]
    z = z[dataSelection]

    tasksToDo = 1
    tasksDone = 0
    minimum   = -np.inf
    params    = "n/a"

    if writingPermissions:
        outfile = open("parameterValues/NeuralNetworkParameters_" + mode + ".txt", "w")

    print("Starting benchmarking with parameters: ")
    for parameter in parameters:
        try:
            tasksToDo *= len(parameters[parameter])
        except:
            pass
        print(parameter, ":", parameters[parameter])
        if writingPermissions:
            outfile.write(str(parameter) + " : " + str(parameters[parameter]) + "\n")

    if randomSearch:
        for i in range(N):
            score = 0
            hiddenLN            = np.random.choice(parameters["hiddenLN"])
            hiddenNN            = np.random.choice(parameters["hiddenNN"])
            epochN              = np.random.choice(parameters["epochN"])
            minibatchSize       = np.random.choice(parameters["minibatchSize"])
            # print((parameters["eta"]))
            eta                 = parameters["eta"][np.random.choice(range(len(parameters["eta"])))]
            # exit()
            lmbd                = np.random.choice(parameters["lmbd"])
            alpha               = np.random.choice(parameters["alpha"])
            activationFunction  = np.random.choice(parameters["activationFunction"])
            outputFunction      = np.random.choice(parameters["outputFunction"])

            for _ in range(parameters["#repetitions"]):
                network = NeuralNetwork(hiddenNN = 32,
                            hiddenLN = hiddenLN)
                
                network.giveInput(X, z)
                network.giveParameters(
                    epochN=epochN,
                    minibatchSize=minibatchSize,
                    eta=etaDefinerDefiner(eta[0], eta[1]),
                    lmbd=lmbd,
                    alpha=alpha,
                    activationFunction=activationFunction,
                    outputFunction=outputFunction
                )

                network.getBiasesWeights()
                network.train()

                if mode.lower()=="classification":
                    score += network.score/parameters["#repetitions"]
                elif mode.lower()=="regression":
                    score += network.R2/parameters["#repetitions"]

            paramSTR = "epochN:{}, minibatchSize:{}, eta:{}, lmbd:{}, alpha:{}, activFunct:{}, outFunc:{}, layers:{}, nodes:{}"\
                        .format(epochN, minibatchSize, eta, lmbd, alpha, activationFunction.__name__, outputFunction.__name__, hiddenLN, hiddenNN)
            if score > minimum:
                minimum = score
                paramSTROUT = paramSTR
                params = [hiddenLN, hiddenNN, epochN, minibatchSize, eta, lmbd, alpha, activationFunction, outputFunction]

            print(i, "/", N, "| score: {:.3f}".format(score), "| params:", paramSTR)

        return minimum, params, paramSTROUT




    for hiddenLN in parameters["hiddenLN"]:
        for hiddenNN in parameters["hiddenNN"]:
            for epochN in parameters["epochN"]:
                for minibatchSize in parameters["minibatchSize"]:
                    for eta in parameters["eta"]:
                        for lmbd in parameters["lmbd"]:
                            for alpha in parameters["alpha"]:
                                for activationFunction in parameters["activationFunction"]:
                                    for outputFunction in parameters["outputFunction"]:
                                        score = 0
                                        tasksDone += 1
                                        for _ in range(parameters["#repetitions"]):
                                            network = NeuralNetwork(hiddenNN = 32,
                                                        hiddenLN = hiddenLN)
                                            
                                            network.giveInput(X, z)
                                            network.giveParameters(
                                                epochN=epochN,
                                                minibatchSize=minibatchSize,
                                                eta=etaDefinerDefiner(eta[0], eta[1]),
                                                lmbd=lmbd,
                                                alpha=alpha,
                                                activationFunction=activationFunction,
                                                outputFunction=outputFunction
                                            )

                                            network.getBiasesWeights()
                                            network.train()

                                            if mode.lower()=="classification":
                                                score += network.score/parameters["#repetitions"]
                                            elif mode.lower()=="regression":
                                                score += network.R2/parameters["#repetitions"]

                                        paramSTR = "epochN:{}, minibatchSize:{}, eta:{}, lmbd:{}, alpha:{}, activFunct:{}, outFunc:{}, layers:{}, nodes:{}"\
                                                    .format(epochN, minibatchSize, eta, lmbd, alpha, activationFunction.__name__, outputFunction.__name__, hiddenLN, hiddenNN)
                                        if score > minimum:
                                            minimum = score
                                            paramSTROUT = paramSTR
                                            params = [hiddenLN, hiddenNN, epochN, minibatchSize, eta, lmbd, alpha, activationFunction, outputFunction]

                                        if writingPermissions:
                                            outfile.write(str(score) + " | " + paramSTR + "\n")

                                        print("Task done:", tasksDone, "/", tasksToDo, "| score: {:.3f}".format(score), "| params:", paramSTR)

    if writingPermissions:
        outfile.write("Optimal: " + str(minimum) + " | " + paramSTROUT)
        outfile.close()

    return minimum, params, paramSTROUT




class NN():
    """Neural Network class, regresses and categorizes

    Use: 
        Give dataset with giveInput(), it will sort training and testing set itself.
        Using dataHandler is an option for this, as is done in this file

        Give parameters with giveParameters()

        Train with train()
    """
    def __init__(self, hiddenNN, hiddenLN):
        """
        Args:
            hiddenNN (int): Number of nodes in each hidden layer. All hidden layers have the same amount of nodes
            hiddenLN ([type]): Number of hidden layers.
        """
        self.hiddenLN = hiddenLN
        self.hiddenNN = hiddenNN

        self.X = None
        self.z = None
        self.x = None
        self.y = None

        self.Ninputs        = None
        self.nfeatures      = None
        self.Ncategories    = None

        self.epochN         = None
        self.minibatchSize  = None
        self.minibatchN     = None
        self.eta            = None
        self.lmbd           = None
        self.alpha          = None
        self.bias           = None

        self.activationFunction = None
        self.outputFunction     = None
        self.prediction         = None
        self.predictedLabel     = None
        self.error              = None
        self.score              = None
        self.R2                 = None
        self.MSE                = None

        self.inputWeights    = None
        self.inputBias       = None
        self.weightsHidden   = None
        self.biasHidden      = None
        self.outWeights      = None
        self.outBias         = None
        self.logisticWeights = None



    def giveInput(self, X=None, z=None, scaling=None):
        """Takes input and output values

        Args:
            X (numpy.nparray, optional) : Input data. Defaults to previously given self.X.
            z (numpy.ndarray, optional) : Expected data, used for training. Defaults to previously given self.z.
            scaling (bool, optional)    : Determines whether or not to scale the data. Defaults to False.
        """
        self.X = defaultingFunc(self.X, X, self.X)
        self.z = defaultingFunc(self.z, z, self.z)
        scaling = defaultingFunc(None, scaling, False)

        if scaling:
            if self.X[:,0].any() != 1:
                X = np.empty(self.X.shape[0], self.X.shape[1]+1)
                X[:,1:] = self.X
                X[:,0]  = 1
                self.X = X
            self.X[:,1:] -= np.mean(self.X[:,1:], axis=0)

        self.Ninputs, self.nfeatures = self.X.shape
        self.Ncategories = self.z.shape[1]
        
        self.activationValues = np.zeros((self.hiddenLN, self.Ninputs, self.hiddenNN))


    def giveParameters(self, 
                        epochN=None,
                        minibatchSize=None,
                        bias=None,
                        eta=None,
                        lmbd=None,
                        alpha=None,
                        activationFunction=None,
                        outputFunction=None):
        """Takes parameters for the network

        Args:
            epochN (int, optional): number of times dataset is iterated through. Defaults to 500.
            minibatchSize (int, optional): amount of datapoints used in each iteration. Defaults to 64.
            bias (float, optional): Is added to nodes before applying activation function. Defaults to 1e-02.
            eta (function or float, optional): learning rate, either given as a function that produces a 
                                               generator or a float. Defaults to 1e-06.
            lmbd (float, optional): momentum rate. Defaults to 1e-06.
            alpha (float, optional): parameter used in activation/output function. Defaults to 1e-01.
            activationFunction (function, optional): function used for activating hidden nodes. Defaults to sigmoid.
            modellingFunoutputFunctionction (function, optional): function used for activeting outgoing nodes. Defaults to softMax.
        """
        
        self.epochN         = defaultingFunc(epochN,                self.epochN,        500)
        self.minibatchSize  = defaultingFunc(minibatchSize,         self.minibatchSize, 64)
        self.bias           = defaultingFunc(bias,                  self.bias,          1e-02)
        self.eta            = defaultingFunc(self.defineEta(eta),   self.eta,           self.defineEta(1e-06))
        self.lmbd           = defaultingFunc(lmbd,                  self.lmbd,          1e-06)
        self.alpha          = defaultingFunc(alpha,                 self.alpha,         1e-01)

        self.activationFunction = defaultingFunc(activationFunction, self.activationFunction, sigmoid)
        self.outputFunction     = defaultingFunc(outputFunction,     self.outputFunction,     softMax)

        self.minibatchN = int(self.Ninputs/self.minibatchSize)


    def getBiasesWeights(self):
        """Generates biases and weights
        Assumes all layers have the same number of nodes

        Behaves as logistic regressor special case if self.hiddenLN == 0
        """
        if self.hiddenLN == 0:
            self.logisticWeights = np.random.randn(self.nfeatures, self.Ncategories)

        else:
            self.inputWeights = np.random.randn(self.nfeatures, self.hiddenNN)
            self.inputBias    = np.zeros(self.hiddenNN) + self.bias

            # First layer of input weights is handled by self.inputWeights, so all 
            # instances of welf.weightsHidden is called at one lower layer 
            # e.g. self.weightsHidden[layer-1]
            self.weightsHidden = np.random.randn(self.hiddenLN-1, self.hiddenNN, self.hiddenNN)
            self.biasHidden    = np.zeros((self.hiddenLN, self.hiddenNN)) + self.bias

            self.outWeights = np.random.randn(self.hiddenNN, self.Ncategories)
            self.outBias = np.zeros(self.Ncategories) + self.bias
        

    def defineEta(self, definer, constant=None):
        """Creates eta as a generator

        Args:
            definer (function or int): Either gives a function that will give a generator or a float that will be wrapped in a generator
            constant (float, optional): if definer is given as a float, definer will be divided by constant each iteration. Defaults to None.

        Returns:
            generator: Generates eta values
        """
        if callable(definer):
            eta = definer
        
        elif constant is None:
            def eta():
                while True:
                    yield definer

        else:
            def eta():
                e = constant
                while True:
                    yield e
                    e /= definer
            
        return eta()


    def feedIn(self, X=None):
        """Feeds data into first hidden layer

        Args:
            X (numpy.ndarray, optional): input data. Defaults to previously given data.
        """
        X = defaultingFunc(self.X, X, None)

        self.activationValues[0] = X @ self.inputWeights + self.inputBias

        self.activationValues[0] = self.activationFunction(self.activationValues[0], self.alpha)


    def feedForward(self, layer):
        """Feeds data between hidden layers, from layer-1 to layer

        Args:
            layer (int): layer being fed into
        """
        self.activationValues[layer] = self.activationValues[layer-1] @ self.weightsHidden[layer-1]
        self.activationValues[layer]+= self.biasHidden[layer]
        self.activationValues[layer] = self.activationFunction(self.activationValues[layer], self.alpha)


    def feedOut(self):
        """Feeds data out to the output layer, calculates error, and generates MSE, R2 and Accuracy Score
        """
        self.prediction = self.activationValues[-1] @ self.outWeights + self.outBias
        self.prediction = self.outputFunction(self.prediction, self.alpha)

        self.error = self.prediction - self.z
        self.MSE   = MSE(self.z, self.prediction)
        self.R2    = R2(self.z, self.prediction)
        self.score = accuracyScore(self.z, self.prediction)

        self.predictedLabel = np.argmax(self.prediction, axis=1)


    def logisticSpecialCasePredict(self):
        """If there are 0 hidden layers, the network behaves as a logistic
        regressor, creating this special case
        """
        self.prediction = self.X @ self.logisticWeights
        self.prediction = self.outputFunction(self.prediction, self.alpha)

        self.error = self.prediction - self.z
        self.MSE   = MSE(self.z, self.prediction)
        self.R2    = R2(self.z, self.prediction)
        self.score = accuracyScore(self.z, self.prediction)

        self.predictedLabel = np.argmax(self.prediction, axis=1)


    def logisticSpecialCasePropagateBack(self):
        """If there are 0 hidden layers, the network behaves as a logistic
        regressor, creating this special case
        """
        gradient   =  self.X.T @ ( self.outputFunction( self.prediction /self.minibatchSize ) - self.z ) 
        self.logisticWeights -= next(self.eta) * gradient


    def predict(self, X=None, z=None):
        """Feeds data through the network and predicts values for self.X.
        If X is given self.z will temporarily be set to an array of correct
        shape for the prediction with all elements equal to 0

        Args:
            X (numpy.ndarray, optional): If given, will be used to predict by the network. Defaults to self.X.
            z (numpy.ndarray, optional): If given, will be used to judge the prediction done. Defaults to self.z
                                         If X is given as none, z-argument will be discarded
        """
            

        if X is not None:
            X_storage = self.X
            z_storage = self.z
            self.X = X
            self.z = defaultingFunc(None, z, np.zeros((X.shape[0], self.Ncategories)))
            self.giveInput()

        if self.hiddenLN == 0:
            self.logisticSpecialCasePredict()
        else:
            self.feedIn()
            for i in range(1, self.hiddenLN):
                self.feedForward(i)
            self.feedOut()

        if X is not None:
            self.X = X_storage
            self.z = z_storage


    def backPropagate(self):
        """Propagates error and gradients backwards through the network and adjusts the weights
        """
        # Propagates error from output layer into the last hidden layer and 
        # adjusts output weights
        outWeightsGradient  = self.activationValues[-1].T @ self.error
        outBiasGradient     = np.sum(self.error,axis=0)
        if self.lmbd > 0.0:
            outWeightsGradient += self.lmbd * self.outWeights
        self.outWeights -= next(self.eta) * outWeightsGradient
        self.outBias    -= next(self.eta) * outBiasGradient


        errorHidden = self.error @ self.outWeights.T * self.activationFunction(self.activationValues[-1], self.alpha, derivative=True)
        for layer in range(self.hiddenLN-1, 0, -1):
            # The output layer is not modelled under self.weightsHidden, so 
            # there is a special case for the last layer
            if layer != self.hiddenLN - 1:

                errorHidden = errorHidden @ self.weightsHidden[layer].T 
                errorHidden*= self.activationFunction(self.activationValues[layer], self.alpha, derivative=True)

            # hiddenWeightGradient = next(self.eta) * errorHidden.T @ self.activationValues[layer-1]/self.Ninputs
            hiddenWeightGradient = next(self.eta) * self.activationValues[layer-1].T @ errorHidden/self.Ninputs
            hiddenBiasGradient   = next(self.eta) * np.sum(errorHidden, axis=0)/self.Ninputs

            if self.lmbd > 0.0:
                hiddenWeightGradient += self.lmbd * self.weightsHidden[layer-1]

            self.weightsHidden[layer-1] -= hiddenWeightGradient
            self.biasHidden[layer]    -= hiddenBiasGradient

        # Propagates error from second layer into the first hidden layer and 
        # adjusts input weights
        inputWeightsGradient = self.X.T @ errorHidden
        inputBiasGradient    = np.sum(errorHidden, axis=0)
        if self.lmbd > 0.0:
            inputWeightsGradient += self.lmbd * self.inputWeights
        self.inputWeights -= next(self.eta) * inputWeightsGradient/self.Ninputs
        self.inputBias    -= next(self.eta) * inputBiasGradient/self.Ninputs

        
    def train(self, splitData=None):
        """Trains network, ignores numpy warnings to simplify training with
        multiple parameters where some weights will explode

        Splits dataset into training and testing data (80%20%, respectively) if splitData is True

        Only trains on batches of size self.minibatchSize at a time

        After training, if splitData is True, the network is fed one last time 
        with test data to generate a score.

        Args:
            splitData (boolean, optional): Will split data into training and testing sets if True, won't otherwise. Defaults to True.
        """
        splitData = defaultingFunc(None, splitData, True)

        if self.inputWeights is None and self.logisticWeights is None: self.getBiasesWeights()
        X_full, z_full = self.X, self.z

        if splitData:
            X_train, X_test, z_train, z_test = train_test_split(self.X, self.z)
        else:
            X_train = X_full
            z_train = z_full

        batchIndexSelectionPool = range(X_train.shape[0])
        batchSelection = np.random.choice(batchIndexSelectionPool, self.minibatchSize, replace=False)

        self.X, self.z = X_train[batchSelection,:], z_train[batchSelection,:]
        self.giveInput()

        self.scoreHistory = np.zeros((self.epochN, 2))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for epoch in range(self.epochN):
                for _ in range(self.minibatchN):
                    batchSelection = np.random.choice(batchIndexSelectionPool, self.minibatchSize, replace=False)
                    self.X, self.z = X_train[batchSelection,:], z_train[batchSelection,:]
                    self.predict()
                    if self.hiddenLN != 0:
                        self.backPropagate()
                    else:
                        self.logisticSpecialCasePropagateBack()
                self.scoreHistory[epoch] = self.R2, self.score
            else:
                if splitData:
                    self.X, self.z = X_test, z_test
                    self.giveInput()
                    self.predict()

        self.X, self.z = X_full, z_full





if __name__ == "__main__":
    from dataHandler import dataHandler

    # CLASSIFICATION

    hiddenLayers        = 0
    hiddenNeurons       = 32
    epochN              = 1000
    minibatchSize       = 32
    eta                 = (1, 1000000)
    lmbd                = 1e-01
    alpha               = 1e-00
    activationFunction  = ReLU_leaky
    outputFunction      = softMax

    handler = dataHandler()
    handler.makeImageLabel()
    X, z = handler.X, handler.z

    
    network = NN(hiddenNN = hiddenNeurons,
                 hiddenLN = hiddenLayers)
    network.giveInput(X, z)
    network.giveParameters(
        epochN=epochN,
        minibatchSize=minibatchSize,
        eta=etaDefinerDefiner(eta[0], eta[1]),
        lmbd=lmbd,
        alpha=alpha,
        activationFunction=activationFunction,
        outputFunction=outputFunction
    )
    network.train()

    paramSTR = "epochN:{}, minibatchSize:{}, eta:{}, lmbd:{}, alpha:{}, activFunct:{}, outFunc:{}"\
                .format(epochN, minibatchSize, eta, lmbd, alpha, activationFunction.__name__, outputFunction.__name__)    

    print("Manually chosen parameters; classification:\n   Layers: {}, Neurons: {} | score (1 is good): {}  | params: {}\n"\
          .format(hiddenLayers, hiddenNeurons, network.score, paramSTR))
    
    handler.printPredictions(network)





    # REGRESSION

    handler.craftDataset(N=32)
    handler.makePolynomial(p=3)
    X, z = handler.X, handler.z

    hiddenLayers        = 1
    hiddenNeurons       = 32
    epochN              = 5000
    minibatchSize       = 32
    eta                 = 1e-01
    lmbd                = 1e-04
    alpha               = 1e-01
    activationFunction  = ReLU_leaky
    outputFunction      = linearActivation

    network = NN(hiddenNN = hiddenNeurons,
                 hiddenLN = hiddenLayers)
    network.giveInput(X, z)
    network.giveParameters(
        epochN=epochN,
        minibatchSize=minibatchSize,
        eta=eta,
        lmbd=lmbd,
        alpha=alpha,
        activationFunction=activationFunction,
        outputFunction=outputFunction
    )

    network.train()
    paramSTR = "epochN:{}, minibatchSize:{}, eta:{}, lmbd:{}, alpha:{}, activFunct:{}, outFunc:{}"\
                .format(epochN, minibatchSize, eta, lmbd, alpha, activationFunction.__name__, outputFunction.__name__)   

    print("Manually chosen parameters; regression:\n   Layers: {}, Neurons: {} | score (1 is good): {}  | params: {}\n"\
          .format(hiddenLayers, hiddenNeurons, network.R2, paramSTR))
    
    handler.plotRealAgainstPredicted(network)



    # Make false not to run parametrization
    parametrizeClassification = False
    parametrizeRegression     = False
    writeToFile = False


    # Parametrization for classification - start
    if parametrizeClassification: 
        parameters = {
            "hiddenLN":             [0,1,2],
            "hiddenNN":             [16, 32],
            "epochN":               [250, 500, 750],
            "minibatchSize":        [32, 64, 128],
            "eta":                  [[j, i**k] for i in np.logspace(0, 6, 7) for j, k in [(1, 1), (None, -1)]],
            "lmbd":                 np.logspace(-1, -6, 3),
            "alpha":                np.logspace(-0, -1, 2),
            "activationFunction":   [sigmoid, ReLU_leaky, ReLU],
            "outputFunction":       [softMax],

            "#repetitions": 5,
            "datafraction": 0.3
        }

        handler = dataHandler()
        handler.makeImageLabel()
        X, z = handler.X, handler.z

        optimalScore, optimalParams, optimalParamSTR = benchmark(X, z, parameters, NN, mode="classification", writingPermissions=writeToFile)

        print(optimalScore, optimalParamSTR)

        network = NN(hiddenNN = optimalParams[1], 
                        hiddenLN = optimalParams[0])
        network.giveInput(X, z)
        network.giveParameters(
            epochN=optimalParams[2],
            minibatchSize=optimalParams[3],
            eta=etaDefinerDefiner(optimalParams[4][0], optimalParams[4][1]),
            lmbd=optimalParams[5],
            alpha=optimalParams[6],
            activationFunction=optimalParams[7],
            outputFunction=optimalParams[8]
        )
        network.getBiasesWeights()
        network.train()

        print("Layers: {}, Neurons: {} | score (1 is good): {} | params: {}".format(optimalParams[0], optimalParams[1], network.score, optimalParamSTR))
        handler.printPredictions(network)

    # Parametrization for classification - end



    # Parametrization for regression - start
    if parametrizeRegression: 
        parameters = {
            "hiddenLN":             [0, 1, 2, 4],
            "hiddenNN":             [16, 32, 48, 64],
            "epochN":               [250, 500],
            "minibatchSize":        [32, 64],
            "eta":                  [[j, i**k] for i in np.logspace(0, 6, 7) for j, k in [(1,1), (None,-1)]],
            "lmbd":                 np.logspace(-1, -6, 6),
            "alpha":                np.logspace(-1, -3, 3),
            "activationFunction":   [ReLU, sigmoid, ReLU_leaky],
            "outputFunction":       [linearActivation],

            "#repetitions": 5,
            "datafraction": 0.3
        }

        handler.craftDataset(N=32)
        handler.makePolynomial(p=15)
        X, z = handler.X, handler.z

        optimalScore, optimalParams, optimalParamSTR = benchmark(X, z, parameters, NN, mode="regression", writingPermissions=writeToFile)

        print(optimalScore, optimalParamSTR)

        network = NN(hiddenNN = optimalParams[1], 
                        hiddenLN = optimalParams[0])
        network.giveInput(X, z)
        network.giveParameters(
            epochN=optimalParams[2],
            minibatchSize=optimalParams[3],
            eta=etaDefinerDefiner(optimalParams[4][0], optimalParams[4][1]),
            lmbd=optimalParams[5],
            alpha=optimalParams[6],
            activationFunction=optimalParams[7],
            outputFunction=optimalParams[8]
        )
        network.getBiasesWeights()
        network.train()

        print("Layers: {}, Neurons: {} | score (1 is good): {} | params: {}".format(optimalParams[0], optimalParams[1], network.R2, optimalParamSTR))
        handler.plotRealAgainstPredicted(network)

    # Parametrization for regression - end