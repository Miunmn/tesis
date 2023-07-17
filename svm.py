from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def run_svm():
  ### Datasets folder
  dataset_folder = './content/sfp-datasets/'
  os.chdir(dataset_folder)
  datasets = os.listdir()
  total_report = ""
  
  print('files in the folder: ', datasets)

  for dataset in datasets:
    data_df = pd.read_csv(dataset)
    faulty_modules = data_df[data_df['defects'] == True]
    non_fauty_modules = data_df[data_df['defects'] == False]
    print("Number of faulty modules: ", len(faulty_modules))
    print("Number of non faulty modules: ", len(non_fauty_modules))

    ## Selected Columns
    feature_df = data_df[['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment',
                        'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']]
    X = np.asarray(feature_df)
    Y = np.asarray(data_df['defects'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 4)
    from sklearn import svm

    # kernels = ["linear", "poly", "rbf", "sigmoid"]
    # gammas = [0.1, 1, 100]
    # Cs = [0.1, 1, 100]
    # for kernel in kernels:
    #   for gamma in gammas:
    #     for c in Cs:
    #       classifier = svm.SVC(kernel=kernel, gamma=gamma, C=c, class_weight='balanced')
    #       classifier.fit(X_train, y_train)
    #       y_predict = classifier.predict(X_test)
    #       report = f"Dataset: {dataset} {kernel} {gamma} {c}\n"
    #       report += classification_report(y_test, y_predict)
    #       report += "--------------------------------------------------\n\n"
    #       print("report", report)
    #       total_report += report
    # ir probando hiperparametros aqui para ver cual es el mejor
    classifier = svm.SVC(kernel='linear', gamma=1, C=10, class_weight='balanced')
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    report = f"Dataset: {dataset} \n"
    report += classification_report(y_test, y_predict)
    print("output_dict_format", classification_report(y_test, y_predict, output_dict=True))
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)

    # Precision Score
    precision_score = confusion_matrix[0][0] / (confusion_matrix[0][1] + confusion_matrix[0][0] )

    # Recall Score
    recall_score = confusion_matrix[0][0] / (confusion_matrix[1][0] + confusion_matrix[0][0] )

    # F1 Score
    f1_score = 2*precision_score*recall_score / (precision_score + recall_score)

    print(f"Precision Score: {precision_score}")
    print(f"Recall Score: {recall_score}")
    print(f"F1 Score: {f1_score}")
    
    report += "--------------------------------------------------\n\n"
    print("report", report)
    total_report += report
  os.chdir('../../')
  

  with open('SVM-LINEAR', 'w') as f:
    f.write(total_report)
