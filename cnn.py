import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


CONTENT_FOLDER = os.path.join(Path(__file__).parent, 'content')

def run_cnn():
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
    X = data_df.iloc[:, :-1].values
    y = data_df.iloc[:, -1].values

    # Codificamos las etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

    # X_train = tf.expand_dims(X_train, axis=0)
    # y_train = tf.expand_dims(y_train, axis=0)
    # y_test = tf.expand_dims(y_test, axis=0)
    # X_test = tf.expand_dims(X_test, axis=0)

    print(X_train.shape[0])
    # Train Model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=["accuracy"])
    model.summary()

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print('Pérdida en el conjunto de prueba:', loss)
    print('Precisión en el conjunto de prueba:', accuracy)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
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
    sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="Blues_r") 
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.show()