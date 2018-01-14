import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Read the data


train_data = pd.read_csv("train_data.csv")
train_data_clean = train_data[['age_group_5_years', 'race_eth', 'first_degree_hx',
       'age_menarche', 'age_first_birth', 'BIRADS_breast_density',
       'current_hrt', 'menopaus', 'bmi_group', 'biophx']]

test_data = pd.read_csv("test_data.csv")
test_data_clean = test_data[['age_group_5_years', 'race_eth', 'first_degree_hx',
       'age_menarche', 'age_first_birth', 'BIRADS_breast_density',
       'current_hrt', 'menopaus', 'bmi_group', 'biophx']]

def read_train_labels():
    labels_data = []
    with (open('train-y','r')) as master_data:
        for y in master_data:
            labels_data.append(int(y.strip()))
    return(np.array(labels_data))

# One hot encoding

def oneHotEnconde(df):
    enc = OneHotEncoder()
    df_1=pd.DataFrame()
    for col in df.columns.values:
        df[col] = df[col].astype(str)
        if df[col].dtypes == 'object':
            enc.fit(df[[col]].values)
            temp = enc.transform(df[[col]])
            temp = pd.DataFrame(temp.toarray(), columns=[(col+"_"+str(i)) for i in df[col].value_counts().index])
            temp = temp.set_index(df.index.values)
            df_1 = pd.concat([df_1,temp],axis=1)
    return df_1


train_data_clean = oneHotEnconde(train_data_clean)
features = train_data_clean.values
labels_data = read_train_labels()

X_train, X_val, y_train, y_val = train_test_split(features, labels_data, test_size=0.2, random_state=1)

test_data_clean = oneHotEnconde(test_data_clean)

# Random Forest Classifier

clf = RandomForestClassifier(n_jobs=1000, random_state=1, max_features=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print("The accuracy on the valid ation data for Random Forest classifier:", accuracy_score(y_pred, y_val))
test_pred = clf.predict(test_data_clean.values)
print("Predictions for Random Forest classifier:\n", test_pred, "\n\n")


# Logistic Regression classifier

clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_val)
print("The accuracy on the valid ation data for Logistic Regression classifier: ", accuracy_score(y_pred, y_val))
test_pred = clf2.predict(test_data_clean.values)
print("Predictions on the test data for the Logistic Regression classifier:\n", test_pred, "\n\n")


# Support Vector Machine Classifier

clf3 = svm.LinearSVC()
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_val)
print("The accuracy on the valid ation data for SVM classifier:", accuracy_score(y_pred, y_val))
test_pred = clf3.predict(test_data_clean.values)
print("Predictions on the test data for the SVM classifier:\n", test_pred, "\n\n")

# Decision Tree classifier


clf4 = DecisionTreeClassifier()
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_val)
print("The accuracy on the valid ation data for the Decision tree classifier:", accuracy_score(y_pred, y_val))
test_pred = clf4.predict(test_data_clean.values)
print("Predictions on the test data for the Decision Tree classifier:\n", test_pred, "\n\n")

