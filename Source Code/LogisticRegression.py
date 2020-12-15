import numpy as np
import math  as mt
from matplotlib.pyplot import plot, axis, xlabel, ylabel, title, show, legend, scatter
from activationFunctions import softMax
from miscFunctions import to_categorical_numpy, defaultingFunc, accuracyScore, MSE, R2
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import SGDRegressor

def benchmark(X, z, parameters, writeToFile=False):
    if writeToFile:
        outfile = open("parameterValues/LogisticRegressionParameters.txt", "w")
        outfile.write("values: | Score | (t0, t1) | # of epochs | Minibatch size | Momentum-coefficient | l2\n\n\n")

    tasks = 1
    for i in parameters:
        if i == "repetitions":
            pass
        else:
            tasks *= len(parameters[i])
    # tasks = len(steps)*len(epochs)*len(batchSize)*len(dragcoeffs)*len(l2params)

    tasksDone = 0
    bestScore = 0
    bestParams = "n/a"
    for t0, t1 in parameters["steps"]:
        for epochN in parameters["epochs"]:
            for batch in parameters["batchSize"]:
                for drag in parameters["dragcoeffs"]:
                    for l2 in parameters["l2params"]:
                        score = 0
                        params = [[t0,t1], epochN, batch, drag, l2]

                        for _ in range(parameters["repetitions"]):
                            Xtr, Xte, ztr, zte = train_test_split(X, z)
                            regressor = LogisticClassifier()
                            regressor.makeImageLabel(batch, epochN)
                            regressor.defineStepLength(t0, t1)
                            regressor.drag = drag
                            regressor.l2 = l2
                            regressor.X = Xtr
                            regressor.z = ztr
                            regressor.N, regressor.nfeatures = Xtr.shape
                            regressor.minibatchN = int(regressor.N/regressor.minibatchSize)
                            regressor.classify()

                            score += regressor.predict(Xte, zte) / parameters["repetitions"]

                        if writeToFile:
                            outfile.write(str(score) + ", " + str(params) + "\n")

                        if  score > bestScore:
                            bestScore = regressor.ACCscore
                            bestParams = params
                        
                        tasksDone += 1
                        print(tasksDone, "/", tasks, " |  ", score, ",", params)

    if writeToFile:
        outfile.write("\n\nOptimal score | parameters: " + str(bestScore) + " " + str(bestParams))
        outfile.close()


    print("\n\nAND THE BEST SCORE ISSSSS:")
    print("   ", bestScore, bestParams)



class LogisticClassifier():
    """
    Logistic Regression Classifier
    """

    def __init__(self, polydegree=None):

        self.N              = None
        self.minibatchSize  = None
        self.minibatchN     = None
        self.epochN         = None

        self.z      = None
        self.theta  = None
        self.X      = None

        self.nfeatures      = None
        self.noisefactor    = None

        self.stepLength         = None
        self.drag               = None
        self.l2                 = None
        self.modellingFunction  = None
        self.outputFunction     = None

        self.ACCscore   = None


    def FrankeFunction(self, x,y):
        """Function modelled in the by the SGD class
        The Franke Function is two-dimensional and is comprised of four 
        natural exponents

        Args:
            x (np.ndarray): array of floats between 0 and 1. x-dimension
            y (np.ndarray): array of floats between 0 and 1. y-dimension

        Returns:
            (numpy.ndarray) : resultant array from Frankefunction
        """
        term1 = 0.75*np.exp(-(9*x-2)**2/4.00 - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-(9*x+1)**2/49.0 - 0.10*(9*y+1))
        term3 = 0.50*np.exp(-(9*x-7)**2/4.00 - 0.25*((9*y-3)**2))
        term4 =-0.20*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4
        

    def defineStepLength(self, t0=None, t1=None):
        """
        stepLength = lambda t: t0/t+t1

        Args:
            t0 (float, optional): nominator. Defaults to 10.
            t1 (float, optional): added with t to denominator. Defaults to 50.
        """
        if t0 is None: t0 = 10
        if t1 is None: t1 = 10
        self.stepLength = lambda t: t0/(t+t1)


    def makeImageLabel(self, minibatchSize=None, epochN=None, outputFunction=None):
        """Imports dataset from sklearn.datasets.load_digits().

        Args:
            minibatchSize (int, optional): Number of datapoints calculated at a time. Defaults to 64.
            epochN (int, optional): Number of times dataset is iterated through. Defaults to 500.
            outputFunction (function, optional) : Function input data is put through
        """
        self.outputFunction = defaultingFunc(self.outputFunction, outputFunction, softMax)
        self.minibatchSize  = defaultingFunc(self.minibatchSize,  minibatchSize,  64)
        self.epochN         = defaultingFunc(self.epochN,         epochN,         500)

        from sklearn import datasets
        digits = datasets.load_digits()
        digits.images = digits.images; digits.target = digits.target
        images = digits.images
        labels = digits.target
        images = images.reshape(len(images), -1)
        labels = labels.reshape(len(labels), -1)
        labels = to_categorical_numpy(labels)

        self.X = images
        self.z = labels

        self.N, self.nfeatures = self.X.shape
        self.minibatchN = int(self.N/self.minibatchSize)


    def classify(self):
        """The main Stochastic Gradient Descent method

        Generates a random theta using numpy.random.rand(self.nfeatures,1) and 
        adjusts it using the SGD method 

        Will call makeImageLabel() if self.X is not yet defined and defineStepLength() 
        if self.stepLength is not yet defined, using default values for each
        """
        if self.X is None: 
            print("in LogisticClassifier.classify(): using defaulting self.makeImageLabel() due to self.X not being defined")
            regressor.makeImageLabel()

        if self.stepLength is None: 
            print("in LogisticClassifier.classify(): using defaulting self.defineStepLength() due to self.stepLength not being defined")
            self.defineStepLength()
            
        self.theta = np.random.rand(self.nfeatures, self.z.shape[1]) 

        indexSelectionPool = range(self.N)
        inertia = np.zeros_like(self.theta)
        drag = defaultingFunc(self.drag, self.drag, 0)
        l2 = defaultingFunc(self.l2, self.l2, 0)

        for epoch in range(self.epochN):
            batch = np.random.choice(indexSelectionPool, self.minibatchSize, replace=False)
            for batchIndex in range(self.minibatchN):

                xi = self.X[batch].reshape(self.minibatchSize,-1)
                zi = self.z[batch].reshape(self.minibatchSize,-1)
                # .reshape gets correct shape if self.minibatchSize = 1, which
                #  might result in floats instead of arrays when using the @ operator

                error = self.outputFunction( xi @ self.theta /self.minibatchSize ) - zi
                gradient = xi.T @ error

                step = self.stepLength(epoch*self.minibatchN + batchIndex)
                
                l2_term = (self.theta**2).sum() * l2
                inertia = drag*inertia
                gradient += inertia + l2_term

                self.theta -= gradient*step


        # self.theta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z
        # Analytical solution that can be used to compare results

        prediction = self.X @ self.theta

        self.ACCscore = accuracyScore(self.outputFunction(prediction/self.N), self.z)


    def predict(self, X, z):
        self.prediction = X @ self.theta
        self.predictedLabel = np.argmax(self.prediction, axis=1)
        return  accuracyScore(self.outputFunction(self.prediction/self.N), z)




    
if __name__ == "__main__":

    regressor = LogisticClassifier()
    regressor.makeImageLabel(minibatchSize  = 128, 
                            epochN         = 1000)

    regressor.defineStepLength(t0=1, t1=32)
    regressor.drag = 0.8
    regressor.l2 = 0
    regressor.classify()
    print(regressor.ACCscore)


    testParams  = True
    writeToFile = False

    if testParams:
        benchmark(writeToFile)
        # if writeToFile:
        #     outfile = open("parameterValues/LogisticRegressionParameters.txt", "w")
        #     outfile.write("values: | Score | (t0, t1) | # of epochs | Minibatch size | Momentum-coefficient | l2\n\n\n")

        # steps       = [[1,i] for i in np.logspace(0,2,5)]
        # epochs      = [50, 100, 500, 1000]
        # batchSize   = [16,32,64,128]
        # dragcoeffs  = np.linspace(0,1,11)
        # l2params    = np.concatenate((np.logspace(1,-3,5), [0]))
        # tasks = len(steps)*len(epochs)*len(batchSize)*len(dragcoeffs)*len(l2params)
        # tasksDone = 0
        # bestScore = 0
        # bestParams = "n/a"
        # for t0, t1 in steps:
        #     for epochN in epochs:
        #         for batch in batchSize:
        #             for drag in dragcoeffs:
        #                 for l2 in l2params:
        #                     regressor = LogisticClassifier()
        #                     regressor.makeImageLabel(batch, epochN)
        #                     regressor.defineStepLength(t0, t1)
        #                     regressor.drag = drag
        #                     regressor.l2 = l2
        #                     regressor.classify()

        #                     score = regressor.ACCscore
        #                     params = [[t0,t1], epochN, batch, drag, l2]
        #                     if writeToFile:
        #                         outfile.write(str(score) + ", " + str(params) + "\n")

        #                     if  score > bestScore:
        #                         bestScore = regressor.ACCscore
        #                         bestParams = params
                            
        #                     tasksDone += 1
        #                     print(tasksDone, "/", tasks, " |  ", score, ",", params)

        # if writeToFile:
        #     outfile.write("\n\nOptimal score | parameters: " + str(bestScore) + " " + str(bestParams))
        #     outfile.close()


        # print("\n\nAND THE BEST SCORE ISSSSS:")
        # print("   ", bestScore, bestParams)

