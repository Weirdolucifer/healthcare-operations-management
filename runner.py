import pandas as pd
import numpy as np
import models
import xlsxwriter

def run(my_sheet, modelList, writer):
    
    # loading the facility data
    df = pd.read_pickle('data/'+my_sheet+'.pkl')

    # Optimal subset selection process 
    corr = df.corr()

    columns = np.full((corr.shape[0],), False, dtype=bool)
    i = 0
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.17:
            columns[j] = True

    ParameterList = list(df.columns[columns])
   
    # Data Splitting
    dataX = df.iloc[:, 1:]
    dataY = df.iloc[:, 0]

    matSize = len(modelList)
    R2 = np.zeros((matSize,1));        MAE = np.zeros((matSize,1));        MSE = np.zeros((matSize,1))
    R2_std = np.zeros((matSize,1));    MAE_std = np.zeros((matSize,1));    MSE_std = np.zeros((matSize,1))
    trainR2 = np.zeros((matSize,1));   trainMAE = np.zeros((matSize,1));   trainMSE = np.zeros((matSize,1))
    U05 = np.zeros((matSize,1));       U10 = np.zeros((matSize,1));        ModelDes = np.empty([matSize, 1], dtype="U500")
    for d in range(matSize):
        ModelDes[d] = ""

    maxIter = 6
    j = 0

    for model in modelList:
        roundR2 = np.zeros((maxIter,1));        roundMAE = np.zeros((maxIter,1));       roundMSE = np.zeros((maxIter,1))
        roundtrainR2 = np.zeros((maxIter,1));   roundtrainMAE = np.zeros((maxIter,1));  roundtrainMSE = np.zeros((maxIter,1))
        roundU05 = np.zeros((maxIter,1));       roundU10 = np.zeros((maxIter,1))
        
        modelName = ""
        for i in range(maxIter):
            Error, TrainError, ModelDescription = models.main(dataX, dataY, ParameterList, model)
            if i == 0:
                modelName = ModelDescription
                print("Model Name: {}".format(ModelDescription))
            print("For Iteraton {}:".format(i+1))
            print("Train Error: {}".format(TrainError))
            print("Error: {}".format(Error))   
            if i == maxIter-1:
                print("\n")

            roundR2[i] = max(0, Error['R2']);           roundMAE[i] = Error['MAE'];            roundMSE[i] = Error['MSE']
            roundtrainR2[i] = max(0, TrainError['R2']); roundtrainMAE[i] = TrainError['MAE'];  roundtrainMSE[i] = TrainError['MSE']
            roundU05[i] = Error['U05'];                 roundU10[i] = Error['U10']

        R2[j] = round(np.median(roundR2), 4);            MAE[j] = round(np.median(roundMAE),4);            MSE[j] = round(np.median(roundMSE), 4)
        if model >= 16:
            R2[j] = round(np.max(roundR2), 4);  MAE[j] = round(np.min(roundMAE), 4);    MSE[j] = round(np.min(roundMSE), 4)
        R2_std[j] = round(np.std(roundR2), 4);           MAE_std[j] = round(np.std(roundMAE), 4);          MSE_std[j] = round(np.std(roundMSE), 4)
        trainR2[j] = round(np.median(roundtrainR2), 4);  trainMAE[j] = round(np.median(roundtrainMAE), 4); trainMSE[j] = round(np.median(roundtrainMSE), 4)
        U05[j] = round(np.mean(roundU05), 4);            U10[j] = round(np.mean(roundU10), 4)
        ModelDes[j] = modelName 
        j = j + 1 

    ## convert your array into a dataframe
    df = pd.DataFrame({'Model Name': ModelDes[:,0], 'R2' : R2[:,0], 'MAE': MAE[:,0], 'MSE': MSE[:,0], 'R2_std': R2_std[:,0], 'MAE_std': MAE_std[:,0], 'MSE_std': MSE_std[:,0], 'trainR2': trainR2[:,0], 'trainMAE': trainMAE[:,0], 'trainMSE' : trainMSE[:,0], 'U05': U05[:,0], 'U10': U10[:,0]})

    ## save to xlsx file
    df.to_excel(writer, sheet_name = my_sheet, index = False)

if __name__=='__main__':

    modelList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    facility = ['F1', 'F2', 'F3', 'F4']
    path = 'Results.xlsx'
    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
    for my_sheet in facility:
        run(my_sheet, modelList, writer)
    writer.save()