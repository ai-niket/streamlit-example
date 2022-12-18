import streamlit as st
import numpy as np
import pandas as pd
import sklearn.metrics as mat

def test_and_print_metrics(y, y_pred, average, labels):
    """Print the accuracy, precision, recall, f1 score and confusion matrix of the model"""
    print('Test Accuracy: ', mat.accuracy_score(y, y_pred))
    print('Test Precision: ', mat.precision_score(y, y_pred, average=average))
    print('Test Recall: ', mat.recall_score(y, y_pred, average=average))
    print('Test F1 Score: ', mat.f1_score(y, y_pred, average=average))
    conf = mat.confusion_matrix(y, y_pred)
    # use confusion matrix display from sklearn
    disp = mat.ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
    st.pyplot(disp.plot())

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    dataframe