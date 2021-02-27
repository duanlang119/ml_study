# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# data preprocessing:
def data_prep():
    # Importing the dataset
    data = pd.read_csv('data/bank_churn_data.csv', delimiter=',')
    X = data.iloc[:, 3:13]
    y = data.iloc[:, 13]
    geography=pd.get_dummies(X["Geography"],drop_first=True)
    gender=pd.get_dummies(X['Gender'],drop_first=True)
    X=pd.concat([X,geography,gender],axis=1)
    X=X.drop(['Geography','Gender'],axis=1)
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_prep()

classifier1=pd.read_pickle('classifier.pkl')

from lime import lime_tabular

interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification'
)
print(X_test.iloc[88])
print('prediction: {}'.format(classifier1.predict(X_test.iloc[[88],:])))

feature_importance = pd.DataFrame()
feature_importance['variable'] = X_train.columns
feature_importance['importance'] = classifier1.feature_importances_

# feature_importance values in descending order
ft_imp=feature_importance.sort_values(by='importance', ascending=False).head(10)
print(ft_imp)

exp = interpretor.explain_instance(
    data_row=X_test.iloc[88], ##new data
    predict_fn=classifier1.predict_proba
)

rs=exp.as_list()

print(rs)