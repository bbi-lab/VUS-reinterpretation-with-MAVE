#Naive Bayes classifier for varaint effect predictions based on multiplexed functional assays
#

#Imports:
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict
import csv
import os.path

#Path to output folder:
#
save_to_path = 'USER DEFINED PATH'

#Step 1: load data:
#load data sets:
#
full_clinvar_train = pd.read_csv("TP53_full_clinvar.csv")
All_variant_set = pd.read_csv("TP53_all_scores_no_domain.csv")
Ambry_VUS_set = pd.read_csv("TP53_Ambry_VUS_scores_no_domain.csv")
Ambry_absent_train_set = pd.read_csv("TP53_Ambry_train_removed.csv")

#From trainig set, split features "X" and classifications "Y":
#
X = full_clinvar_train.iloc[:,1:-1].values
Y = full_clinvar_train.iloc[:,5].values
variants = full_clinvar_train.iloc[:,0].values

#From test set, split features and classifications:
#
X_all = All_variant_set.iloc[:,1:5].values
variants_all = All_variant_set.iloc[:,0].values

#VUS set for classification:
#From Ambry VUS, split features "X_VUS", classifications "Y_VUS", and varinat "VUS":
#
X_VUS = Ambry_VUS_set.iloc[:,1:-1].values
Y_VUS = Ambry_VUS_set.iloc[:,5].values
VUS = Ambry_VUS_set.iloc[:,0].values

#Ambry set
#
X_ambry = Ambry_absent_train_set.iloc[:,1:-1].values
Y_ambry = Ambry_absent_train_set.iloc[:,5].values
variant_ambry = Ambry_absent_train_set.iloc[:,0].values

#Step 2: process data to test accuracy of different classifier subsets:
#To evaluate classifier vs. any single feature input, train on each feature individually:
#
#Dominant negative reporter assay:
X1 = full_clinvar_train.iloc[:,1]
X1 = np.array(X1)
X1 = X1.reshape(-1,1)
#Nutlin-3 domninant negative assay:
X2 = full_clinvar_train.iloc[:,2]
X2 = np.array(X2)
X2 = X2.reshape(-1,1)
#LOF etoposide assay:
X3 = full_clinvar_train.iloc[:,3]
X3 = np.array(X3)
X3 = X3.reshape(-1,1)
#LOF Nutlin-3 assay:
X4 = full_clinvar_train.iloc[:,4]
X4 = np.array(X4)
X4 = X4.reshape(-1,1)


#Step 3: train classifier:
#Train Naive Bayes classifier from sklearn:
#
classifier = GaussianNB(priors=[0.5,0.5])
classifier.fit(X, Y)


#Step 4:leave one out cross validation and accuracy metrics:
#
loocv = model_selection.LeaveOneOut()

#this will perform loocv on the training set and report the probabilites for each class for each variant, the variant, and the true class 
proba = cross_val_predict(classifier, X, Y, cv=loocv, method='predict_proba')
benign_probs = proba[:,0]
path_probs = proba[:,1]
cv_predict = cross_val_predict(classifier, X, Y, cv=loocv, method="predict")

#this will write a .csv file with the loocv results to the file path specified
for a, b, c, d, e in zip([benign_probs], [path_probs], [variants], [cv_predict], [Y]):
    cv_output = (a, b, c, d, e)

file_name = 'loocv_output.csv'
loocv_output = os.path.join(save_to_path, file_name)
with open(loocv_output, 'w') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerow(["benign_probs", "path_probs", "variant", "NBC_prediction", "ture_class"])
     for row in zip(*cv_output):
         for i, column in enumerate(row):
             f.write(str(column))

             if i != len(row)-1:
                 f.write(',')
         f.write('\n')


#This will calculate the loocv accuracy for the NB classifier and each feature individually:
#4 feature classifier:
acc = accuracy_score(cv_predict, Y)
acc= ("4 feature loocv accuracy:", acc*100)

#Dominant negative reporter only:
cv_predict_DN_reporter = cross_val_predict(classifier, X1, Y, cv=loocv, method="predict")
acc_DN_reporter = accuracy_score(cv_predict_DN_reporter, Y)
acc_DN_reporter = ("DN reporter loocv accuracy:", acc_DN_reporter*100)

#Nutlin-3 dominant negative only
cv_predict_DN_nutlin = cross_val_predict(classifier, X2, Y, cv=loocv, method="predict")
acc_DN_nutlin = accuracy_score(cv_predict_DN_nutlin, Y)
acc_DN_nutlin = ("Nutlin-3 DN loocv accuracy:", acc_DN_nutlin*100)

#LOF Etoposide only
cv_predict_LOF_eto = cross_val_predict(classifier, X3, Y, cv=loocv, method="predict")
acc_LOF_eto = accuracy_score(cv_predict_LOF_eto, Y)
acc_LOF_eto = ("LOF etoposide loocv accuracy:", acc_LOF_eto*100)

#LOF nutlin-3 only
cv_predict_LOF_nutlin = cross_val_predict(classifier, X4, Y, cv=loocv, method="predict")
acc_LOF_nutlin = accuracy_score(cv_predict_LOF_nutlin, Y)
acc_LOF_nutlin = ("Nutlin-3 LOF loocv accuracy:", acc_LOF_nutlin*100)

#write the output of accuracy analysis to a text file:
acc_file = 'accuracy.txt'
acc_output = os.path.join(save_to_path, acc_file)
acc_text = (str(acc), str(acc_DN_reporter), str(acc_DN_nutlin), str(acc_LOF_eto), str(acc_LOF_nutlin))
with open(acc_output, 'w') as f:
    for line in acc_text:
        f.writelines(line)
        f.writelines('\n')

   
#Step 5: check concordance between classifier and Ambry variants not in training set:
#
ambry_predictions = classifier.predict(X_ambry)
ambry_probs = classifier.predict_proba(X_ambry)
ambry_benign = ambry_probs[:,0]
ambry_path = ambry_probs[:,1]

for a,b,c,d,e in zip([variant_ambry], [ambry_predictions], [ambry_benign], [ambry_path], [Y_ambry]):
    ambry_predict = (a, b, c, d, e)
    
classifier_amb = 'classifier_ambry_predictions.csv'
amb_output = os.path.join(save_to_path, classifier_amb)
with open(amb_output, 'w') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerow(["variant", "prediction", "benign_prob", "path_prob", "ambry_class"])
     for row in zip(*ambry_predict):
         for i, column in enumerate(row):
             f.write(str(column))

             if i != len(row)-1:
                 f.write(',')
         f.write('\n')


#Step 6: perfrom classifications:
#
#This uses the classifier trained on clinvar variants and makes predictions for all variants with multiplexed functional data:
all_predictions = classifier.predict(X_all)
all_probs = classifier.predict_proba(X_all)
all_benign = all_probs[:,0]
all_path = all_probs[:,1]

for a,b,c,d in zip([variants_all], [all_predictions], [all_benign], [all_path]):
    all_predict = (a, b, c, d)

classifier_all = 'classifier_all_predictions.csv'
all_output = os.path.join(save_to_path, classifier_all)
with open(all_output, 'w') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerow(["variant", "prediction", "benign_prob", "path_prob"])
     for row in zip(*all_predict):
         for i, column in enumerate(row):
             f.write(str(column))

             if i != len(row)-1:
                 f.write(',')
         f.write('\n')

#This uses the classifier trained on clinvar variants and makes predictions for Ambry VUS:
VUS_predictions = classifier.predict(X_VUS)
VUS_probs = classifier.predict_proba(X_VUS)
VUS_benign = VUS_probs[:,0]
VUS_path = VUS_probs[:,1]

for a,b,c,d in zip([VUS], [VUS_predictions], [VUS_benign], [VUS_path]):
    VUS_predict = (a, b, c, d)

classifier_VUS = 'classifier_VUS_predictions.csv'
VUS_output = os.path.join(save_to_path, classifier_VUS)
with open(VUS_output, 'w') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerow(["variant", "prediction", "benign_prob", "path_prob"])
     for row in zip(*VUS_predict):
         for i, column in enumerate(row):
             f.write(str(column))

             if i != len(row)-1:
                 f.write(',')
         f.write('\n')


#Step 7: PR curves to compare 4 feature classifier with singe feature classifiers
Y_prediction = cross_val_predict(classifier, X, Y, cv=loocv)
Y_proba_P = cross_val_predict(classifier, X, Y, cv=loocv, method='predict_proba')

#PR for Benign
precision_B, recall_B, _ = precision_recall_curve(Y, Y_proba_P[:,0], pos_label='B')
plt.plot(recall_B, precision_B, color='r',
             label=r'All feature PR B (AUC = %0.2f)' % (average_precision_score(Y, Y_proba_P[:,0], pos_label='B')),
             lw=2, alpha=.8)


#classifier cv on DN reporter
Y_prediction_1 = cross_val_predict(classifier, X1, Y, cv=loocv)
Y_proba_P1 = cross_val_predict(classifier, X1, Y, cv=loocv, method='predict_proba')

#PR for Benign
precision_B1, recall_B1, _ = precision_recall_curve(Y, Y_proba_P1[:,0], pos_label='B')
plt.plot(recall_B1, precision_B1, color='g',
             label=r'DN reporter PR B (AUC = %0.2f)' % (average_precision_score(Y, Y_proba_P1[:,0], pos_label='B')),
             lw=2, alpha=.8)

#classifier cv on DN Nutlin-3
Y_prediction = cross_val_predict(classifier, X2, Y, cv=loocv)
Y_proba_P2 = cross_val_predict(classifier, X2, Y, cv=loocv, method='predict_proba')

#PR for Benign
precision_B2, recall_B2, _ = precision_recall_curve(Y, Y_proba_P2[:,0], pos_label='B')
plt.plot(recall_B2, precision_B2, color='darkorange',
             label=r'DN nutlin-3 PR B (AUC = %0.2f)' % (average_precision_score(Y, Y_proba_P2[:,0], pos_label='B')),
             lw=2, alpha=.8)

#classifier cv on LOF etoposide
Y_prediction = cross_val_predict(classifier, X3, Y, cv=loocv)
Y_proba_P3 = cross_val_predict(classifier, X3, Y, cv=loocv, method='predict_proba')

precision_B3, recall_B3, _ = precision_recall_curve(Y, Y_proba_P3[:,0], pos_label='B')
plt.plot(recall_B3, precision_B3, color='c',
             label=r'LOF etoposide PR B (AUC = %0.2f)' % (average_precision_score(Y, Y_proba_P3[:,0], pos_label='B')),
             lw=2, alpha=.8)

#classifier cv on LOF nutlin-3
Y_prediction = cross_val_predict(classifier, X4, Y, cv=loocv)
Y_proba_P4 = cross_val_predict(classifier, X4, Y, cv=loocv, method='predict_proba')

#PR for Benign
precision_B4, recall_B4, _ = precision_recall_curve(Y, Y_proba_P4[:,0], pos_label='B')
plt.plot(recall_B4, precision_B4, color='gray',
             label=r'LOF nutlin-3 PR B (AUC = %0.2f)' % (average_precision_score(Y, Y_proba_P3[:,0], pos_label='B')),
             lw=2, alpha=.8)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
title="Cross validation PR curve"
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(title)
plt.legend(loc="lower right")
PR_curve_path = save_to_path + "loocv_PR_curve.pdf"
plt.savefig(PR_curve_path)