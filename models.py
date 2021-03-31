import pandas as pd
import numpy as np
from sklearn import linear_model, naive_bayes, tree, ensemble, neural_network
from sklearn.model_selection import train_test_split

def ComputeError(Ypredicted, Y):
    Err = dict()
    Residual = Ypredicted - Y
    
    # R2
    Err['R2'] =  round(1.0 - (np.std(Residual)**2) / (np.std(Y)**2), 4)

    # Mean absolute error
    absR = abs(Residual)
    Err['MAE'] = round(np.mean(absR), 4)

    # Mean absolute squared error
    Err['MSE'] = round(np.mean(Residual**2), 4)

    # Percent of errors under 5 and 10 minutes
    Err['U05'] = round(absR[absR<5].count() / absR.count(), 4)
    Err['U10'] = round(absR[absR<10].count() / absR.count(), 4)
    return Err

def ComputeModel(X_train, X_test, y_train, y_test, ParameterList, modelNum, ModelDescription):
    # Model 1 (Linear regression on most recent wait)
    if modelNum == 1:
        ParameterList = ['MostRecent1']
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X_train[ParameterList], y_train)

        Error = ComputeError(model.predict(X_test[ParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[ParameterList]), y_train)
        ModelDescription = "Linear regression on most recent wait"

    # Model 2 (Linear regression on moving average most recent waits)
    elif modelNum == 2:
        ParameterList = ['MostRecent1', 'MostRecent2', 'MostRecent3', 'MostRecent4', 'MostRecent5']
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X_train[ParameterList], y_train)

        Error = ComputeError(model.predict(X_test[ParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[ParameterList]), y_train)
        ModelDescription = "Linear regression on moving average most recent waits"

    # Model 3 (Linear regression on moving average based on Queue length)
    elif modelNum == 3:
        ParameterList = ['LineCount0', 'LineCount1', 'LineCount2', 'LineCount3', 'LineCount4']
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X_train[ParameterList], y_train)

        Error = ComputeError(model.predict(X_test[ParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[ParameterList]), y_train)
        ModelDescription = "Linear regression on moving average based on Queue length"

    # Model 4 (Linear regression on best features with intercept)
    elif modelNum == 4:
        model = linear_model.LinearRegression(fit_intercept=True)
        AvailableParameterList =  X_train.columns[X_train.columns.isin(ParameterList)]
        model.fit(X_train[AvailableParameterList], y_train)

        Error = ComputeError(model.predict(X_test[AvailableParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[AvailableParameterList]), y_train)
        ModelDescription = "Linear regression on best features with intercept"

    # Model 5 (Linear regression using all attributes without intercept)
    elif modelNum == 5:
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Linear regression using all attributes without intercept"

    # Model 6 (Linear regression using all attributes with intercept)
    elif modelNum == 6:
        model = linear_model.LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Linear regression using all attributes with intercept"

    # Model 7 (Linear weighted regression using all attributes with intercept)
    elif modelNum == 7:
        weights = np.ones(y_train.shape)
        model = linear_model.LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train, weights)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Linear weighted regression using all attributes with intercept"
        

    # Model 8 (Linear Robust regression (TheilSenRegressor) using all attributes with intercept)
    elif modelNum == 8:
        model = linear_model.TheilSenRegressor(fit_intercept=True, n_jobs = -1)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Linear Robust regression (TheilSenRegressor) using all attributes with intercept"

    # Model 9 (Linear Ridge regression (Optimized) using all attributes with intercept)
    elif modelNum == 9:
        model = linear_model.Ridge(fit_intercept=True, normalize=True)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Linear Ridge regression (Optimized) using all attributes with intercept"

    # Model 10 (ElasticNet regression using all attributes with intercept)
    elif modelNum == 10:
        model = linear_model.LassoCV(fit_intercept=True, normalize=True, max_iter= 5000, n_jobs = -1)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "ElasticNet regression using all attributes with intercept"

    # Model 11 (Gaussian naive bayes using all attributes)
    elif modelNum == 11:
        model = naive_bayes.GaussianNB()
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "Gaussian naive bayes using all attributes"

    # Model 12 (Decision Tree)
    elif modelNum == 12:
        AvailableParameterList =  X_train.columns[X_train.columns.isin(ParameterList)]
        model = tree.DecisionTreeClassifier()
        model.fit(X_train[AvailableParameterList], y_train)

        Error = ComputeError(model.predict(X_test[AvailableParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[AvailableParameterList]), y_train)
        ModelDescription = "Decision Tree"

    # Model 13 (RandomForestRegressor with 10 splits)
    elif modelNum == 13:
        model = ensemble.RandomForestRegressor(n_estimators = 10, n_jobs= -1)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "RandomForestRegressor with 10 splits"

    # Model 14 (RandomForestRegressor with 20 splits)
    elif modelNum == 14:
        model = ensemble.RandomForestRegressor(n_estimators = 20, n_jobs= -1)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "RandomForestRegressor with 20 splits"

    # Model 15 (RandomForestRegressor with 30 splits)
    elif modelNum == 15:
        model = ensemble.RandomForestRegressor(n_estimators = 30, n_jobs= -1)
        model.fit(X_train, y_train)

        Error = ComputeError(model.predict(X_test), y_test)
        TrainError = ComputeError(model.predict(X_train), y_train)
        ModelDescription = "RandomForestRegressor with 30 splits"

    # Model 16 (Neural Network single Layer)
    elif modelNum == 16:
        AvailableParameterList =  X_train.columns[X_train.columns.isin(ParameterList)]
        model = neural_network.MLPRegressor(hidden_layer_sizes=(10,))
        model.fit(X_train[AvailableParameterList], y_train)

        Error = ComputeError(model.predict(X_test[AvailableParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[AvailableParameterList]), y_train)
        ModelDescription = "Neural Network single Layer"

    # Model 17 (Neural Network multiple Layer)
    else:
        layer0 = len(ParameterList)

        AvailableParameterList =  X_train.columns[X_train.columns.isin(ParameterList)]
        model = neural_network.MLPRegressor(hidden_layer_sizes=(layer0*2, layer0*4, layer0*2, layer0//2,))
        model.fit(X_train[AvailableParameterList], y_train)

        Error = ComputeError(model.predict(X_test[AvailableParameterList]), y_test)
        TrainError = ComputeError(model.predict(X_train[AvailableParameterList]), y_train)
        ModelDescription = "Neural Network multiple Layer"
    
    return Error, TrainError, ModelDescription

def main(dataX, dataY, ParameterList, modelNum):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    ModelDescription = ""
    Error, TrainError, ModelDescription = ComputeModel(X_train, X_test, y_train, y_test, ParameterList, modelNum, ModelDescription)
    return Error, TrainError, ModelDescription