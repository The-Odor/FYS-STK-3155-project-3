# Source:           https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
# Notebook of note: https://www.kaggle.com/ericswright/predicting-heart-failure-hyperparameter-tuning
from NeuralNetwork import NN, benchmark
benchmarkNN = benchmark
# import NeuralNetwork.benchmark as benchmarkNN
from LogisticRegression import LogisticClassifier, benchmark
benchmarkLog = benchmark
# import LogisticRegression.benchmark as benchmarkLog
import csv
import numpy as np
from activationFunctions import *
from miscFunctions import etaDefinerDefiner
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def confusionMatrix(pred, true, printing=True):
    """Generates a 2x2 confusion matrix for a set of predictions and true values

    Args:
        pred (np.ndarray): Set of predictions. Must be of same 1st dimension as true.
        true (np.ndarray): Set of expected values. Must be of same 1st dimension as pred.
        printing (bool, optional): Will print out the cells of the matrix if True, will not otherwise. Defaults to True.

    Returns:
        Tuple: Returns the confusion matrix as a tuple for further use.
               Will be in order of True Positive, True Negative, False Positive, FalseNegative.
    """
    if len(true.shape) == 2:
        true = true[:,0]
    if len(pred.shape) == 2:
        pred = pred[:,0]

    trueMask  = (true == pred)
    falseMask = (true != pred)

    truePositive = sum(np.logical_and(pred==1, trueMask))
    trueNegative = sum(np.logical_and(pred==0, trueMask))
    falsePositive = sum(np.logical_and(pred==1, falseMask))
    falseNegative = sum(np.logical_and(pred==0, falseMask))
    if printing:
        print("True Positive: ", truePositive)
        print("True Negative: ", trueNegative)
        print("False Positive: ", falsePositive)
        print("False Negative: ", falseNegative)
    return truePositive, trueNegative, falsePositive, falseNegative


def precisionRecall(pred, true, printing=True):
    """Extends confusionMatrix() by calculating positive detection rate (precision) and
    negative detection rate (recall).

    Args:
        pred (np.ndarray): Set of predictions. Must be of same 1st dimension as true.
        true (np.ndarray): Set of expected values. Must be of same 1st dimension as pred.
        printing (bool, optional): Will print out precision and recall if True, will not otherwise. Defaults to True.

    Returns:
        Tuple: Returns the precision and recall as a tuple for further use.
               Will be in order of precision, recall.
    """
    truePositive, trueNegative, falsePositive, falseNegative = confusionMatrix(pred, true, printing=False)
    
    positivesDetected = truePositive / (truePositive + falseNegative)
    negativesDetected = trueNegative / (trueNegative + falsePositive)

    if printing:
        print("Rate of positives detected (Sensitivity)", positivesDetected)
        print("Rate of negatives detected (Specificity)", negativesDetected)

    return positivesDetected, negativesDetected


def featureResponseCorrelation(dataset, dataType, isInputValue):
    """Uses data from getData() to represent individual correlation between input- and response features
    as histograms and tables

    Args:
        dataset (np.ndarray): Data as readied by getData().
        dataType (dict): Dictionary containing datatypes for features as defined by getData().
        isInputValue dict: Dictionary containing definitions for input v response variables.
    """
    # Assumes response variable is in one-hot form

    response = {i:dataset[i].astype(bool) for i in dataset if not isInputValue[i]}
    plots = {i:[] for i in dataset if not isInputValue[i]}

    for key in dataType:
        if dataType[key] == bool:            
            dataT = dataset[key + "_true"]
            dataF = dataset[key + "_false"]
            correlations = np.zeros(2*len(response))
            # print("Correlation table for {}".format(key))
            for i, res in enumerate(response):
                correlations[2*i]   = sum(np.logical_and(dataT, response[res]))
                correlations[2*i+1] = sum(np.logical_and(dataF, response[res]))

                # print("{} : {}: {:.2f}".format(key + "_true",  res, sum(np.logical_and(dataT, response[res]))))
                # print("{} : {}: {:.2f}".format(key + "_false", res, sum(np.logical_and(dataF, response[res]))))


            # print("LaTeX formatting for boolean response variable:")
            print(r"""
\begin{{table}}[H] 
\centering
\caption{{Table of correlation between boolean {Key_clean} and DEATH{{\_}}EVENT}}
\label{{tab:{Key_norm}}}
\begin{{tabular}}{{|c||c|c|}}
    \hline
       & {Key_clean} True   & {Key_clean} False   \\
    \hline
    \hline
    DEATH{{\_}}EVENT True & {TruePos:.2f}  & {FalsePos:.2f} \\
    \hline
    DEATH{{\_}}EVENT False & {FalseNeg:.2f} & {TrueNeg:.2f}  \\
    \hline
    \hline
    DEATH{{\_}}EVENT False & & \\
    divided by & {DEToverDEFfeatureTrue:.2f} & {DEToverDEFfeatureFalse:.2f} \\
    DEATH{{\_}}EVENT True & & \\
    \hline
\end{{tabular}}
\end{{table}}
                """.format(
                    Key_clean = key.replace("_", "{\_}"),
                    Key_norm  = key,
                    TruePos = correlations[0],
                    TrueNeg = correlations[3],
                    FalsePos = correlations[1],
                    FalseNeg = correlations[2],
                    DEToverDEFfeatureTrue = correlations[2] / correlations[0],
                    DEToverDEFfeatureFalse = correlations[3] / correlations[1]
                )
            )


        else:
            data = dataset[key]
            for res in response:
                plots[res] = data[response[res]]

            for res in response:
                plt.hist(plots[res], histtype="step", label=res, density=True)
            plt.title("Spread for the {} feature for each response variable".format(key))
            plt.legend()
            plt.savefig("../LaTeX/images/{}.png".format(key))
            plt.show()
            print(r"""
\begin{{figure}}[H]
        \centering 
        \includegraphics[scale=0.6]{{images/{NormKey}.png}}
        \caption{{Histogram presenting the {CleanKey} feature for DEATH{{\_}}EVENT being True and False}}
        \label{{fig:{NormKey}}}
\end{{figure}}
            """.format(
                CleanKey = key.replace("_", "{\_}"),
                NormKey  = key
                )
            )

        


def getData():
    """Imports and readies the data for use.

    Returns:
        (np.ndarray, dict, dict): (Readied data, datatype for each feature, whether a feature is an input or response variable)
    """
    with open("heart_failure_clinical_records_dataset.csv", "r") as infile:
        infile = csv.reader(infile, delimiter=",")

        # Extracting column names
        # columnNames = next(infile)
        data = {name:[] for name in next(infile)}

        # Extracting data
        for row in infile:
            for column, dataPoint in zip(data.keys(), row):
                data[column].append(dataPoint)

    # Datatypes in this dataset are limited to int, bool, and float. 
    # Sex is a binary value and represented as a bool.
    # Bools are represented in one-hot form
    dataTypes = {
        "age":                          float,
        "anaemia":                      bool,
        "creatinine_phosphokinase":     int,
        "diabetes":                     bool,
        "ejection_fraction":            int,
        "high_blood_pressure":          bool,
        "platelets":                    float,
        "serum_creatinine":             float,
        "serum_sodium":                 int,
        "sex":                          bool, 
        "smoking":                      bool,
        "DEATH_EVENT":                  bool
    }

    isInputValue = {
        "age":                          True,
        "anaemia":                      True,
        "anaemia_true":                 True,
        "anaemia_false":                True,
        "creatinine_phosphokinase":     True,
        "diabetes":                     True,
        "diabetes_true":                True,
        "diabetes_false":               True,
        "ejection_fraction":            True,
        "high_blood_pressure":          True,
        "high_blood_pressure_true":     True,
        "high_blood_pressure_false":    True,
        "platelets":                    True,
        "serum_creatinine":             True,
        "serum_sodium":                 True,
        "sex":                          True, 
        "sex_true":                     True, 
        "sex_false":                    True, 
        "smoking":                      True,
        "smoking_true":                 True,
        "smoking_false":                True,
        "DEATH_EVENT":                  False,
        "DEATH_EVENT_true":             False,
        "DEATH_EVENT_false":            False
    }

    del data["time"]
    # Irrelevant for prediction; a response variable with artifically high predictive power

    # Converting input to arrays
    for column in data:
        data[column] = np.array(data[column])

    # Converting boolean values into one-hot form
    booleanValues = np.array(list(data.keys()))[np.array(list(dataTypes.values())) == bool]
    linearValues  = np.array(list(data.keys()))[np.array(list(dataTypes.values())) != bool]

    for key in booleanValues:
        data[key + "_true"] = data[key].astype(int)
        data[key + "_false"] = np.logical_not(data[key].astype(int)).astype(int)
        del data[key]

    # Converting remaining values to correct datatype
    for key in linearValues:
        data[key] = data[key].astype(dataTypes[key])

    # Scales data
    for key in data:
        if key.endswith("true"):
            continue
        if key.endswith("false"):
            continue
        if dataTypes[key] == bool:
            continue
        data[key] = data[key] / np.average(data[key])

    return data, dataTypes, isInputValue


def randomForest(X, z, test=False):
    """Wrapper for scikit-learns random forest algorithm. Trains a random forest using X and z.

    Args:
        X (np.ndarray): Input data the forest is to be trained on.
        z (np.ndarray): Response data the forest is to be trained against.
        test (bool, optional): If true, will search a hard-coded parameter-
                               space for optimal parameters instead of 
                               training a forest. Defaults to False.

    Returns:
        (float, list): (score reached, [testing set prediction, testing set])
    """
    if not test:
        clf = RandomForestClassifier(
            bootstrap   = True,
            max_depth   = 8,
            max_features    = "auto",
            min_samples_leaf    = 4,
            min_samples_split   = 5,
            n_estimators    = 160
        )

        Xtr, Xte, ztr, zte = train_test_split(X, z)
        clf.fit(Xtr, ztr)
        forestPrediction = clf.predict(Xte)
        score = sum(np.argmax(forestPrediction, axis=1)==np.argmax(zte, axis=1))/len(forestPrediction[:,0])
        return score, [np.argmax(forestPrediction, axis=1), zte]

    else:
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        df = df.sample(frac=1).reset_index(drop=True)
        df=df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
            'serum_creatinine', 'serum_sodium', 'anaemia', 'diabetes',
            'high_blood_pressure','sex','smoking', 'DEATH_EVENT']]
        X_panda = df.iloc[:,:]
        X_corr = X_panda.corr()
        mask = np.zeros_like(X_corr)
        mask[np.triu_indices_from(mask)] = True
        # sns.heatmap(X_corr, annot=True, fmt='.2f', mask=mask)
        # plt.show()
        param_grid = {
            'bootstrap': [True],
            'max_depth': np.arange(6,11,2),
            'max_features': ['auto','sqrt'],
            'min_samples_leaf': np.arange(3,6,2),
            'min_samples_split': np.arange(4,7,2),
            'n_estimators': np.arange(100,351,20)
        }
        from sklearn.model_selection import GridSearchCV
        forest_search=GridSearchCV (
            RandomForestClassifier(),
            param_grid,
            cv=5,
            verbose=False,
            refit=True,
            n_jobs=-1
        )
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        optimalScore = 0
        reps = 5
        print(reps, "tasks to perform. Performed: ", end="")
        for _ in range(reps):
            X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.1)
            forest_clf = make_pipeline(StandardScaler(),forest_search)
            forest_clf.fit(X_train,z_train)
            forestPrediction = forest_clf.predict(X_test)
            optimalScore += (sum(np.argmax(forestPrediction, axis=1)==np.argmax(z_test, axis=1))/len(forestPrediction[:,0])) / reps
            print("|", end="")
        print()
        optimalParam = forest_search.best_params_
        # print("Forest prediction using trained parameters:", )
        print("Optimal randomForest parameters:", optimalScore, optimalParam, sep="\n", end="\n\n")


def NeuralNetwork(X, z, test=False):
    """Wrapper for a neural network. Trains a neural network using X and z.

    Args:
        X (np.ndarray): Input data the network is to be trained on.
        z (np.ndarray): Response data the network is to be trained against.
        test (bool, optional): If true, will search a hard-coded parameter-
                               space for optimal parameters instead of 
                               training a network. Defaults to False.

    Returns:
        (float, list): (score reached, [testing set prediction, testing set])
    """
    if not test:
        hiddenLayers        = 2
        hiddenNeurons       = 64
        epochN              = 500
        minibatchSize       = 32
        eta                 = (None, 1e-03)
        lmbd                = 1e-06
        alpha               = 1e-00
        activationFunction  = sigmoid
        outputFunction      = softMax

        Xtr, Xte, ztr, zte = train_test_split(X, z)

        network = NN(hiddenNN = hiddenNeurons,
                    hiddenLN = hiddenLayers)
        network.giveInput(Xtr, ztr)
        network.giveParameters(
            epochN          = epochN,
            minibatchSize   = minibatchSize,
            eta             = etaDefinerDefiner(eta[0], eta[1]),
            lmbd            = lmbd,
            alpha           = alpha,
            activationFunction  = activationFunction,
            outputFunction      = outputFunction
        )
        network.train(splitData=False)


        network.predict(Xte, zte)

        return network.score, [network.predictedLabel, zte]


    else:
        # Benchmarking parameters; random search
        parameters = {
            "hiddenLN":             [0, 1, 2, 4],
            "hiddenNN":             [16, 32, 64, 128, 256],
            "epochN":               [500],
            "minibatchSize":        [32, 64],
            "eta":                  [[j, i**k] for i in np.logspace(0, 6, 7) for j, k in [(1, 1), (None, -1)]],
            "lmbd":                 np.logspace(-1, -6, 3),
            "alpha":                np.logspace(-0, -1, 1),
            "activationFunction":   [sigmoid, ReLU_leaky, ReLU],
            "outputFunction":       [softMax],

            "#repetitions": 5,
            "datafraction": 1
        }

        optimalScore, optimalParams, optimalParamSTR = benchmarkNN(X, z, parameters, NN, mode="classification", randomSearch=False, writingPermissions=False, N=int(1e3))
        print("Optimal Neural Network parameters:", optimalScore, optimalParamSTR, sep="\n", end="\n\n")


def ridge(X, z, test=False, lamb=0):
    """Wrapper for ridge algorithm. Trains a ridge regressor using X and z.

    Args:
        X (np.ndarray): Input data the regressor is to be trained on.
        z (np.ndarray): Response data the regressor is to be trained against.
        test (bool, optional): If true, will search a hard-coded parameter-
                               space for optimal parameters instead of 
                               training a regressor. Defaults to False.

    Returns:
        (float, list): (score reached, [testing set prediction, testing set])
    """
    if not test:
        X_train, X_test, y_train, y_test = train_test_split(X, z)
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        beta = np.linalg.pinv(X_train.T @ X_train + lamb*np.identity(X_train.shape[1])) @ X_train.T @ y_train
        ypredict = X_test @ beta
        ypredictLabel = np.argmax(softMax(ypredict), axis=1)
        score = sum(ypredictLabel == np.argmax(y_test, axis=1))/len(y_test) 
        return score, [ypredictLabel, y_test]

    else:
        lambdas = np.linspace(0,1,100)
        reps = 50
        optimalScore = -np.inf
        optimalParam = "n/a"
        for lamb in lambdas:
            score = 0
            for _ in range(reps):
                score += ridge(X, z, test=False, lamb=lamb) / reps
            if score > optimalScore:
                optimalScore = score
                optimalParam = lamb
        print("Optimal Ridge parameters:", optimalScore, optimalParam, sep="\n", end="\n\n")


def LogisticClassification(X, z, test=False):
    """Wrapper for a logistic regressor algorithm. Trains a logistic regressor using X and z.

    Args:
        X (np.ndarray): Input data the regressor is to be trained on.
        z (np.ndarray): Response data the regressor is to be trained against.
        test (bool, optional): If true, will search a hard-coded parameter-
                               space for optimal parameters instead of 
                               training a regressor. Defaults to False.

    Returns:
        (float, list): (score reached, [testing set prediction, testing set])
    """
    if not test:
        t0t1    = (100, 30)
        epochN  = 500
        batch   = 16
        drag    = 0.6
        l2      = 0

        Xtr, Xte, ztr, zte = train_test_split(X, z)

        regressor = LogisticClassifier()
        regressor.makeImageLabel(batch, epochN)
        regressor.defineStepLength(t0=t0t1[0], t1=t0t1[1])
        regressor.drag  = drag
        regressor.l2    = l2
        regressor.X     = Xtr
        regressor.z     = ztr

        regressor.N, regressor.nfeatures = Xtr.shape
        regressor.minibatchN = int(regressor.N/regressor.minibatchSize)

        regressor.classify()

        score = regressor.predict(Xte, zte)

        return score, [regressor.predictedLabel, zte]

    else:
        parameters = {
            "steps"       : [[j,i] for i in np.logspace(1,7,14) for j in np.logspace(1, 7, 7)],
            "epochs"      : [500],
            "batchSize"   : [16,32,64],
            "dragcoeffs"  : np.linspace(0,1,6),
            "l2params"    : np.concatenate((np.logspace(0,-3,4), [0])),
            "repetitions" : 5
        }
        benchmarkLog(X, z, parameters, writeToFile=False)





if __name__ == "__main__":

    # CONFIG START

    # evaluateOverIterations runs each operations testForAverage times and gives the average score.
    evaluateOverIterations      = False

    # If "evaluateOverIterations" is True, testsForAverage determines the.
    # amount of iterations are performed for evaluation.
    testsForAverage = 1000

    # evaluateXParameters runs the given algorithm over a grid-search of parameters and reports the best set.
    evaluateNNParameters        = False
    evaluateRidgeParameters     = False
    evaluateForestParameters    = False
    evaluateLogisticParameters  = False


    # Determines whether to print confusion matrix or the feature correlation tables and histograms.
    # Confusion matrix will also be taken as an average over testsForAverage times
    showConfusionMatrix = False
    showfeatureCorrelation = False # Currently set to save figures as files, see line 149.

    # CONFIG END

    # OPERATIONS START
    data, dataTypes, isInputValue = getData()


    X = np.array([data[feature] for feature in data if isInputValue[feature]]).T
    z = (np.array([data[feature] for feature in data if not isInputValue[feature]]).T).reshape(X.shape[0], -1)


    if evaluateOverIterations:
        print("Training {} times, printing average".format(testsForAverage))
        print("Minimum score due to guessing all numbers being the same:", max((sum(z[:,0])/len(z[:,0]), 1-sum(z[:,0])/len(z[:,0]))))


        # Evaluation of Neural Network
        score = 0
        for i in range(testsForAverage):
            score += NeuralNetwork(X, z)[0] / testsForAverage
        print("Own neural network prediction accuracy:", score)


        # Evaluation of random forest
        score = 0
        for i in range(testsForAverage):
            score += randomForest(X, z)[0] / testsForAverage
        print("Scikit Learn random forest prediction accuracy:", score)


        # Evaluation of ridge
        score = 0
        for i in range(testsForAverage):
            score += ridge(X, z, lamb=0.1)[0] / testsForAverage
        print("Ridge prediction accuracy:", score)

        # Evaluation of Logistic Regression
        score = 0
        for i in range(testsForAverage):
            score += LogisticClassification(X, z)[0] / testsForAverage
        print("Logistic prediction accuracy:", score)


    if showConfusionMatrix:
        methods = {
        "Neural Network"        : NeuralNetwork,
        "Random Forest"         : randomForest,
        "Ridge"                 : ridge,
        "Logistic Classifier"   : LogisticClassification           
        }
        for method in methods:
            confusionArray = np.zeros(4)
            detectionArray = np.zeros(2)
            cumulativeScore = 0
            for i in range(testsForAverage):
                score, predtrue = methods[method](X, z)
                pred, true = predtrue
                # The returned prediction is the predicted label/index, where index=0 is True and index=1 is false
                pred = np.logical_not(pred)

                confusionTable = confusionMatrix(pred, true, printing=False)
                detectionTable = precisionRecall(pred, true, printing=False)

                cumulativeScore += score / testsForAverage
                for i in range(4):
                    confusionArray[i] += confusionTable[i] / testsForAverage
                for i in range(2):
                    detectionArray[i] += detectionTable[i] / testsForAverage

            # print("\n\nConfusion matrix for: {}, with an overall score of {}".format(method, cumulativeScore))
            # print("True Positive: {:.2f}".format(confusionArray[0]))
            # print("True Negative: {:.2f}".format(confusionArray[1]))
            # print("False Positive: {:.2f}".format(confusionArray[2]))
            # print("False Negative: {:.2f}".format(confusionArray[3]))

            # print("\nPositive Detection Rate: {:.2f}".format(detectionArray[0]))
            # print("Positive Detection Rate: {:.2f}".format(detectionArray[1]))

            # print("LaTeX formatting:")

            print(r"""
\begin{{table}}[H] 
\centering
\caption{{Confusion table for {Method} as an average over {TimesRun} runs, with an overall accuracy score of {Score:.2f}{{\%}}}}
\label{{tab:{Method}}}
\begin{{tabular}}{{|c||c|c|}}
\hline
{Method}       & Positive   & Negative   \\
\hline
\hline
Pred. Positive & {TruePos:.2f}  & {FalsePos:.2f} \\
\hline
Pred. Negative & {FalseNeg:.2f} & {TrueNeg:.2f}  \\
\hline
\hline
Rate of Detection & {PositiveDetectionRate:.2f} & {NegativeDetectionRate:.2f} \\
\hline
\end{{tabular}}
\end{{table}}
                """.format(
                    Method = method,
                    TruePos = confusionArray[0],
                    TrueNeg = confusionArray[1],
                    FalsePos = confusionArray[2],
                    FalseNeg = confusionArray[3],
                    PositiveDetectionRate = detectionArray[0],
                    NegativeDetectionRate = detectionArray[1],
                    TimesRun = testsForAverage,
                    Score = cumulativeScore*100
                )
            )

    if showfeatureCorrelation:
        featureResponseCorrelation(data, dataTypes, isInputValue)

    if evaluateNNParameters:
        NeuralNetwork(X, z, test=True)

    if evaluateForestParameters:
        randomForest(X, z, test=True)

    if evaluateRidgeParameters:
        ridge(X, z, test=True)

    if evaluateLogisticParameters:
        LogisticClassification(X, z, test=True)