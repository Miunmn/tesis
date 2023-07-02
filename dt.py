import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


CONTENT_FOLDER = os.path.join(Path(__file__).parent, 'content')

def run_dt():
  dataset_folder = os.path.join(CONTENT_FOLDER, 'sfp-datasets')
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

    # Train Model
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_train, y_train)

    # Testing Model
    test_pred_decision_tree = decision_tree.predict(X_test)
    print("="*4 + " Predicciones " + "="*4)
    print(test_pred_decision_tree)

    confusion_matrix = metrics.confusion_matrix(y_test, test_pred_decision_tree)
    matrix_df = pd.DataFrame(confusion_matrix)

    # Precision Score
    precision_score = confusion_matrix[0][0] / (confusion_matrix[0][1] + confusion_matrix[0][0] )

    # Recall Score
    recall_score = confusion_matrix[0][0] / (confusion_matrix[1][0] + confusion_matrix[0][0] )

    # F1 Score
    f1_score = 2*precision_score*recall_score / (precision_score + recall_score)

    print(f"Precision Score: {precision_score}")
    print(f"Recall Score: {recall_score}")
    print(f"F1 Score: {f1_score}")
    
    ax = plt.axes()
    sns.set(font_scale=1.3)
    sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma") 
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.show()
