
import numpy as np               

from dtaidistance import dtw


x_train_file = np.load('new_train_data.npy')
y_train_file = np.load('new_train_label.npy')

x_train = np.array(x_train_file)
y_train = np.array(y_train_file)

x_test_file = np.load('new_test_data.npy')
y_test_file = np.load('new_test_label.npy')

x_test = np.array(x_test_file)
y_test = np.array(y_test_file)


pred = []
x =0
y_ttest = []
for i in range(0,len(y_test)):
    final = []
    for xtrain in x_train:
        distance = dtw.distance_fast(x_test[i],xtrain)
        final.append(distance)
    x=x+1
    print(x)
    mini = final.index(min(final))
    pred.append(y_train[mini]) 
    y_ttest.append(y_test[i])
    
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_ttest, pred))
print(classification_report(y_ttest, pred))

import pandas as pd
clsf_report = pd.DataFrame(classification_report(y_true = y_ttest, y_pred = pred, output_dict=True)).transpose()
clsf_report.to_csv('Your Classification Report Name.csv', index= True)