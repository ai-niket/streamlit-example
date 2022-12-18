import streamlit as st
import pandas as pd
import sklearn.metrics as mat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def test_and_print_metrics(y, y_pred, average, labels):
    """Print the accuracy, precision, recall, f1 score and confusion matrix of the model"""
    print('Test Accuracy: ', mat.accuracy_score(y, y_pred))
    print('Test Precision: ', mat.precision_score(y, y_pred, average=average))
    print('Test Recall: ', mat.recall_score(y, y_pred, average=average))
    print('Test F1 Score: ', mat.f1_score(y, y_pred, average=average))
    conf = mat.confusion_matrix(y, y_pred)
    # use confusion matrix display from sklearn
    disp = mat.ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
    st.pyplot(disp.plot().figure_)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    test_and_print_metrics(y_test, y_test_pred, 'weighted', knn.classes_)
