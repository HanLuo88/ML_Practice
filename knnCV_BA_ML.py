import numpy as np
from statistics import mean, mode
import pandas as pd
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import warnings
from xgboost.core import Booster

from xgboost.plotting import plot_tree

warnings.filterwarnings("ignore")


# Python Scikit-Learn natives KNN benutzen, um ein geeignetes K zu ermitteln
# Einlesen der CSV-Datei und Data-Preprocessing
pimaDF = pd.read_csv('diabetes.csv')
testresult_to_int = {'tested_negative': 0, 'tested_positive': 1}
pimaDF['class'] = pimaDF['class'].map(testresult_to_int)

pima_class = pimaDF.iloc[:, -1]
pima_features = pimaDF.iloc[:, :-1]

# Aufteilen der Daten in 4 Untersets
pima_features_train, pima_features_test, pima_class_train, pima_class_test = train_test_split(
    pima_features, pima_class, test_size=0.2, random_state=7, stratify=pima_class)


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

# Knn-Classifier for K up to 300
# accarray = []
# for k in range(1,301):

#     pimaKNN = KNeighborsClassifier(n_neighbors=k)
#     #Training
#     pimaKNN.fit(pima_features_train,pima_class_train)
#     #Accuracy
#     acc = pimaKNN.score(pima_features_test,pima_class_test )
#     accarray.append((k,acc))
#print('K: ', k, 'Accuracy: ', acc)
# print(accarray)
# Beste Accuracy war 78%

###########################################################################################################################

# Jetzt mit n-fold Cross-Validation
# acc_mean = []
# for k_value in range(1,301):

#     pimaCV = KNeighborsClassifier(n_neighbors=k_value)
#     #Training mit n-fold
#     cv_scores = cross_val_score(pimaCV,pima_features, pima_class, cv=20)
#     acc_mean.append(np.mean(cv_scores))
#     #print('K: ', k_value,'mean cv_scores: ', np.mean(cv_scores))
# print(acc_mean)
# print(np.max(acc_mean))
# Beste Accuracy war 75%

###########################################################################################################################

# Jetzt mit XGBoost
xgmodel = XGBClassifier(eval_metric='error')
# xgmodel.fit(pima_features_train, pima_class_train)
# print(xgmodel) zeigt die Parameter des Classifiers an

# xgboosted_prediction = xgmodel.predict(pima_features_test)

# acc = accuracy_score(pima_class_test, xgboosted_prediction)
# print("Accuracy with all Features: ", acc)
acc_CV = cross_val_score(xgmodel, pima_features, pima_class, cv=10)
print(acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
# print(mean(acc_CV))
# print(xgmodel.feature_importances_)
# plot_importance(xgmodel)
# pyplot.show()
###########################################################################################################################
###########################################################################################################################
# Von den 8 Features nehmen ich die 6 mit den höchsten F-Scores. Das sind:
#pedi, plas, mass, age, pres, preg
most_important_features = pima_features.loc[:, ['pedi', 'plas', 'mass', 'age', 'pres', 'preg']]
#Reduziertes Dataset splitten
# reduced_pima_features_train, reduced_pima_features_test, reduced_pima_class_train, reduced_pima_class_test = train_test_split(
#     most_important_features, pima_class, test_size=0.2, random_state=7, stratify=pima_class)
xgmodel_reduced_features = XGBClassifier(eval_metric='error')
# print(xgmodel_reduced_features)
# xgmodel_reduced_features.fit(reduced_pima_features_train, reduced_pima_class_train)
# reduced_xgboosted_prediction = xgmodel_reduced_features.predict(reduced_pima_features_test)
# acc_reduced = accuracy_score(reduced_pima_class_test, reduced_xgboosted_prediction)
# print("Accuracy with 6 most relevant Features: ", acc_reduced)
# print(xgmodel_reduced_features.feature_importances_)


# # #Direkt mit CV
accCV_reduced_features = cross_val_score(xgmodel_reduced_features, most_important_features, pima_class, cv=10)
print(accCV_reduced_features, "Mean-Accuracy with the 6 most relevant Features: ", mean(accCV_reduced_features))

