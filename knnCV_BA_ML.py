import warnings
warnings.filterwarnings("ignore")
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

from xgboost.core import Booster

from xgboost.plotting import plot_tree




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

# Knn-Classifier for K up to 200
# accarray = []
# for k in range(1,301):
#     pimaKNN = KNeighborsClassifier(n_neighbors=k)
#     #Training
#     pimaKNN.fit(pima_features_train,pima_class_train)
#     #Accuracy
#     acc = pimaKNN.score(pima_features_test,pima_class_test )
#     accarray.append((k,acc))
#     print('K: ', k, 'Accuracy: ', acc)
# print(accarray)
# Beste Accuracy war 79%

###########################################################################################################################
###########################################################################################################################
# Jetzt mit n-fold Cross-Validation
# acc_mean = []
# for k_value in range(1, 201):
#     pimaCV = KNeighborsClassifier(n_neighbors=k_value)
#     # Training mit n-fold
#     cv_scores = cross_val_score(pimaCV, pima_features, pima_class, cv=10)
#     acc_mean.append(np.mean(cv_scores))
#     print('K with all Features: ', k_value,'           Mean of the cv_scores per K-value: ', np.mean(cv_scores))
#print('Max-Accuracy with all Features: ', max(acc_mean))

# Beste Accuracy war 75%
###########################################################################################################################
###########################################################################################################################

# Jetzt mit XGBoost
# xgmodel = XGBClassifier(eval_metric='error')
# xgmodel.fit(pima_features_train, pima_class_train)
# #print(xgmodel) zeigt die Parameter des Classifiers an
# xgboosted_prediction = xgmodel.predict(pima_features_test)
# acc_CV = cross_val_score(xgmodel, pima_features, pima_class, cv=10)
# print(acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
# print(xgmodel.feature_importances_)
# plot_importance(xgmodel)
# pyplot.show()
###########################################################################################################################
###########################################################################################################################
# Von den 8 Features nehmen ich die 6 mit den höchsten F-Scores. Das sind:
#pedi, plas, mass, age, pres, preg
most_important_features = pima_features.loc[:, [
    'pedi', 'plas', 'mass', 'age', 'pres', 'preg']]
# # Reduziertes Dataset splitten
# reduced_pima_features_train, reduced_pima_features_test, reduced_pima_class_train, reduced_pima_class_test = train_test_split(
#     most_important_features, pima_class, test_size=0.2, random_state=7, stratify=pima_class)
# xgmodel_reduced_features = XGBClassifier(eval_metric='error')
# # print(xgmodel_reduced_features)
# xgmodel_reduced_features.fit(reduced_pima_features_train, reduced_pima_class_train)
# reduced_xgboosted_prediction = xgmodel_reduced_features.predict(reduced_pima_features_test)
# acc_reduced = accuracy_score(reduced_pima_class_test, reduced_xgboosted_prediction)
# print("Accuracy with 6 most relevant Features: ", acc_reduced)

########################################################################################################
########################################################################################################
# # #Direkt mit CV
# acc_reduced = []
# for kneighbors in range(1, 201):
#     xgmodel_reduced_features = KNeighborsClassifier(n_neighbors=kneighbors)
#     # Training mit n-fold
#     cv_scores_reduced = cross_val_score(xgmodel_reduced_features, most_important_features, pima_class, cv=10)
#     acc_reduced.append(np.mean(cv_scores_reduced))
#     print('K with reduced Features: ', kneighbors,'           Mean of the cv_scores per K-value: ', np.mean(cv_scores_reduced))
# accCV_reduced_features = cross_val_score(xgmodel_reduced_features, most_important_features, pima_class, cv=10)
# print("Max-Accuracy with the 6 most relevant Features: ", max(acc_reduced))
########################################################################################################
########################################################################################################

# Mit allen Features war der stabilste Bereich für K von 6 bis 90.
# K-Wert als Wurzel der Anzahl der Datenpunkt zu nehmen ist hier also durchaus sinnvoll und liegt bei 27.
# Mit den 6 laut XGBoost relevantesten Features ist der stabile Bereich der K-Werte im Prinzip von 1 bis 120.
# Auch hier kann man also K = 27 nehmen. Man könnte sogar noch weiter runtergehen um rechenleistung/Zeit zu sparen und K = 3 oder 5 wählen.


# Jetzt mit train_test_split und prediction eines datapunktes bis ein truepositiv gefunden wird(jemand der mit diabetes diagnostiziert wurde
# und auch tatsächlich diabetes hat) und speicher sie in eine Liste
pima_features_train_reduced, pima_features_test_reduced, pima_class_train_reduced, pima_class_test_reduced = train_test_split(
    most_important_features, pima_class, test_size=0.2, random_state=7, stratify=pima_class)

#KNN with reduced features
reduced_pima_model = KNeighborsClassifier(n_neighbors=5)
#Training reduced model
reduced_pima_model.fit(pima_features_train_reduced, pima_class_train_reduced)


reduced_pimaDF = pimaDF.loc[:, [
    'pedi', 'plas', 'mass', 'age', 'pres', 'preg', 'class']]
tmp_class_test_reduced = pima_class_test_reduced.to_numpy()
#True Positivs finden
for i in range(len(pima_features_test_reduced)):
    result = reduced_pima_model.predict(
        pima_features_test_reduced.iloc[i:i+1, :])
    if(result == 1 and result == tmp_class_test_reduced[i]):
        print('i: ', i, '   True Value: ', tmp_class_test_reduced[i], 'result: ', result,
              '    True Positiv', '    Feature_test: ', pima_features_test_reduced.iloc[i:i+1, :])
        break
########################################################################################################
########################################################################################################

testperson_truepositiv = reduced_pimaDF.iloc[748:749, :-1]
actual_person_without_diabetes = reduced_pimaDF.iloc[18:19, :-1]

print('Person 748, die tatsächlich Diabetes hat(True Positiv): ',
      testperson_truepositiv)
tmpdf = reduced_pimaDF.to_numpy()
print('Person 18, die laut Datensatz kein Diabetes hat: ',
      actual_person_without_diabetes)


testperson_truepositiv_array = testperson_truepositiv.to_numpy()
actual_person_without_diabetes_array = actual_person_without_diabetes.to_numpy()

testperson_truepositiv_array[0][1] = actual_person_without_diabetes_array[0][1]
result = reduced_pima_model.predict(testperson_truepositiv_array)
print('spalte: 1', 'Neuer Wert in dieser Spalte: ', testperson_truepositiv_array[0][1], '  Diagnose: ', result)

#Ermitteln, welches Feature des Diabeteskranken durch den Wert eines gesunden Patienten ersetzt werden muss.
# for col in range(0,6):
#     testperson_truepositiv_array[0][col] = actual_person_without_diabetes_array[0][col]
#     result = reduced_pima_model.predict(testperson_truepositiv_array)
#     print('spalte: ', col, '    Neuer Wert in dieser Spalte: ', testperson_truepositiv_array[0][col], '  Diagnose: ', result)
#Hier: Reicht wenn Spalte 1 ersetzt wird
