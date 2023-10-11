import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Reading Source File
SourceFile = pd.read_excel('XRPTrain.xlsx', header=None)  # Reading a Network file
SourceData = SourceFile.to_numpy()
Test_Data = pd.read_excel('XRPTest.xlsx', header=None)
Test_Data = np.array(Test_Data)

# Defining Minimum and maximum values for normalizing
Min_Values = np.min(SourceData, axis=0)
Max_Values = np.max(SourceData, axis=0)

# Normalizing the data
for i in range(SourceData.shape[0]):
    for j in range(SourceData.shape[1]):
        SourceData[i, j] = (SourceData[i, j] - Min_Values[j]) / (Max_Values[j] - Min_Values[j])

# Preparing Data for Neural Network
X_train = SourceData[:, 1:SourceData.shape[1]]
Y_train = SourceData[:, 0]

# Creating Neural Network and training
mlp = MLPRegressor(hidden_layer_sizes=100, activation='tanh', solver='sgd',
                   batch_size=5, learning_rate='adaptive', learning_rate_init=0.001,
                   max_iter=1000, shuffle=True, tol=0.000001, verbose=True, momentum=0.9)
mlp.fit(X_train, Y_train)
print(1, '   ', mlp.score(X_train, Y_train))

# Normalize and predict test data
for i in range(Test_Data.shape[0]):
    for j in range(1, Test_Data.shape[1]):
        if Test_Data[i, j] < Min_Values[j] or Test_Data[i, j] > Max_Values[j]:
            print("Data is out of the range of the trained Network")
        else:
            Test_Data[i, j] = (Test_Data[i, j] - Min_Values[j]) / (Max_Values[j] - Min_Values[j])

X_test = Test_Data[:, 1:Test_Data.shape[1]]
Y_test = Test_Data[:, 0]

# Predicting new Values and De-Normalizing
Y_predicted = mlp.predict(X_test)
Y_predicted = Y_predicted * (Max_Values[0] - Min_Values[0]) + Min_Values[0]

# Plotting the predicted and actual values
plt.plot(Y_predicted[1:50], label="Predicted Data")
plt.plot(Y_test[1:50], label="Actual Values")
plt.legend()
plt.show()
