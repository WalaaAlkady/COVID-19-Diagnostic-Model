#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
#from matplotlib.colors import ListedColormap
#from sklearn.linear_model import LogisticRegression

#Importing the libraries
import numpy as np
import pandas as pd

#preprocessing and feature selection methods
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif , f_classif, f_regression, chi2, RFE, RFECV
from sklearn.decomposition import PCA as sklearnPCA
from random import sample

#training & testing
from sklearn.model_selection import cross_val_score #, ShuffleSplit
from sklearn.model_selection import train_test_split

#Binarization
from sklearn.preprocessing import LabelEncoder , LabelBinarizer, MinMaxScaler, OrdinalEncoder

#classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#performance
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#Importing the dataset:
# CSV file
dataset = pd.read_csv('D:\Master\Covid19\prot\data_processed.csv')
#dataset = pd.read_csv('D:\Master\Covid19\prot\mm2test.csv')

dataset = pd.read_csv('D:\Master\Covid19\prot\mmc2Full.csv')
#remove nan:
dataset1 = dataset.dropna(axis=1)

#x = dataset1.iloc[:,2:793].values
x = dataset1.iloc[:,2:377].values
y = dataset1.iloc[:,1].values

#Preprocessing:
normalized_x = preprocessing.normalize(x)


#Merge testing with training data
z = pd.DataFrame(columns = dataset2.columns)

for col in dataset2.columns:
    if (col in dataset.columns):
        z[col] = dataset[col]
    else:
        z[col] = np.nan

z.to_csv(r'D:\Master\Covid19\prot\mmc2test.csv', index = False)
        
#count elements in each class:
#Array fr will store frequencies of element  
fr = [None] * len(y);  
visited = -1;  
   
for i in range(0, len(y)):  
    count = 1;  
    for j in range(i+1, len(y)):  
        if(y[i] == y[j]):  
            count = count + 1;  
            #To avoid counting same element again  
            fr[j] = visited;  
              
    if(fr[i] != visited):  
        fr[i] = count;  
   
#Displays the frequency of each element present in array  
print("---------------------");  
print(" Element | Frequency");  
print("---------------------");  
for i in range(0, len(fr)):  
    if(fr[i] != visited):  
        print("    " + str(y[i]) + "    |    " + str(fr[i]));  
print("---------------------"); 



#Classifiers:

#KNN:
#binary labels:

KNN = KNeighborsClassifier(n_neighbors=20,metric='minkowski',p=2)
KNN.fit(Xb_train,Yb_train)
Pred = KNN.predict(Xb_test)

accKNN = 0
for i in range (len(Pred)):
    if(all(rec in Pred[i] for rec in Yb_test[i])):
        accKNN += 1
accKNN = (accKNN/len(Pred))*100

acc_KNN = (cm_KNN[0][0]+cm_KNN[1][1])/(cm_KNN[0][0]+cm_KNN[1][1]+cm_KNN[1][0]+cm_KNN[0][1]) * 100

#sensitivity == recall == True Positive Rate
#Specificity == True Negative Rate
TN, FP, FN, TP = cm_KNN.ravel()
# Sensitivity, hit rate, recall, or true positive rate
Sens_KNN = TP/(TP+FN) *100
# Specificity or true negative rate
Spec_KNN = TN/(TN+FP) *100
# Precision or positive predictive value
Prec_KNN = TP/(TP+FP) *100

#Labels:
KNN = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=1)
scores = cross_val_score(KNN, normalized_x, y, cv=10)
KNNacc = scores.mean() * 100
KNNstd = scores.std() * 2

#..................................................................................................

#NB:
#binary labels:
BC = OneVsRestClassifier(GaussianNB())
BC.fit(Xb_train,Yb_train)
Pred = BC.predict(Xb_test)

accNB = 0
for i in range (len(Pred)):
    if(all(rec in Pred[i] for rec in Yb_test[i])):
        accNB += 1
accNB = (accNB/len(Pred))*100

#Labels:
BC = GaussianNB()
scores = cross_val_score(BC, normalized_x, y, cv=10)
BCacc = scores.mean() * 100
BCstd = scores.std() * 2


#..................................................................................................

#SVM:
#binary labels:
svm = OneVsRestClassifier(SVC(kernel='linear',random_state=0))
svm.fit(Xb_train,Yb_train)
Pred = svm.predict(Xb_test)

accSVM = 0
for i in range (len(Pred)):
    if(all(rec in Pred[i] for rec in Yb_test[i])):
            accSVM += 1
accSVM = (accSVM/len(Pred))*100

#Labels:
clf = SVC(kernel='poly')
scores = cross_val_score(clf, x, y, cv=10)
SVMacc = scores.mean() * 100
SVMstd = scores.std() * 2

#..................................................................................................

#NN:
#binary labels:


#Labels:
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
scores = cross_val_score(NN, normalized_x, y, cv=10)
NNacc = scores.mean() * 100
NNstd = scores.std() * 2

#..................................................................................................

#DT:
#binary labels:
DT = DecisionTreeClassifier(random_state=0,criterion="entropy")
DT.fit(Xb_train,Yb_train)
Pred = DT.predict(Xb_test)

accDT = 0
for i in range (len(Pred)):
    if(all(rec in Pred[i] for rec in Yb_test[i])):
        accDT += 1
accDT = (accDT/len(Pred))*100

#Labels:
DT = DecisionTreeClassifier(random_state=0,criterion="entropy")
DT.fit(X_train,Y_train)
Pred = DT.predict(X_test)

accDT = 0
for i in range (len(Pred)):
    if(Pred[i]==Y_test[i]):
        accDT += 1
accDT = (accDT/len(Pred))*100

#..................................................................................................

#RF:
#binary labels:
RF = RandomForestClassifier(n_estimators=10,random_state=0,criterion="entropy")
RF.fit(Xb_train,Yb_train)
Pred = RF.predict(Xb_test)

accRF = 0
for i in range (len(Pred)):
    if(all(rec in Pred[i] for rec in Yb_test[i])):
        accRF += 1
accRF = (accRF/len(Pred))*100

#Labels:
RF = RandomForestClassifier(n_estimators=200,random_state=0,criterion="entropy")
scores = cross_val_score(RF, normalized_x, y, cv=10)
RFacc = scores.mean() * 100
RFstd = scores.std() * 2


#..................................................................................................
#..................................................................................................

#scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')

'''
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
n_samples = x.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
'''
