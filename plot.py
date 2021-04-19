import matplotlib.pyplot as plt

def waitplot(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_pred_g = y_pred[::70]
    y_test_g = y_test[::70]

    fig = plt.figure()
    if max(y_test_g) >= max(y_pred_g):
        my_range = int(max(y_test_g))
    else:
        my_range = int(max(y_pred_g))
    plt.plot(range(len(y_test_g)), y_test_g, 'b-')
    plt.plot(range(len(y_pred_g)), y_pred_g, 'g-')
    plt.xlabel('Appointments DataPoint')
    plt.ylabel('Waiting Time (Mins)')
    fig.legend(labels = ('Actual Waiting Time','Predicted Waiting Time'),loc='upper center')
    plt.savefig('3.png')


def Probabilityplot(model, X_test, X_train):
    y_pred = model.predict(X_test)

    ParameterList = ['ThoracicCount']
    AvailableParameterList =  X_train.columns[X_train.columns.isin(ParameterList)]
    X_test_facility = X_test[AvailableParameterList]
    X_test_facility = X_test_facility[:300]
    X_test_facility = np.where(X_test_facility > 6, 1, 0)
    y_pred_facility = y_pred[:300]
    y_pred_facility = abs(y_pred_facility) / max(abs(y_pred_facility)) 

    fig = plt.figure()
    if max(X_test_facility) >= max(y_pred_facility):
        my_range = int(max(X_test_facility))
    else:
        my_range = int(max(y_pred_facility))
    
    plt.plot(range(len(y_pred_facility)), y_pred_facility, 'b-')

    k = 0
    for i in range(30):
        c = 0
        for j in range(10):
            if X_test_facility[k] == 1:
                c = c+1
            k = k + 1
        if c > 3:
            k = k - 10
            for jk in range(10):
                X_test_facility[k] = 1
                k = k+1

    for i in range(len(X_test_facility)):
        if X_test_facility[i] == 1:
            plt.axvline(x =i, color=(0, 0, 0, 0.20))

    plt.xlabel('Time (Hours)')
    plt.ylabel('Probability of getting a queue in Thoracic test')
    fig.legend(labels = ('Probability(Actual queue length) > 6',' Probability (Predicted queue length)'),loc='upper center')
    plt.savefig('2.png')