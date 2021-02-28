import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# data preprocessing:
def data_prep():
    # Importing the dataset
    data = pd.read_csv('data/bank_churn_data.csv', delimiter=',')
    # Drop the columns as explained above
    # df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    X = data.iloc[:, 3:13]
    y = data.iloc[:, 13]
    
    # replace one hot coding
    geography=pd.get_dummies(X["Geography"],drop_first=True)
    gender=pd.get_dummies(X['Gender'],drop_first=True)
    X = pd.concat([X, geography, gender], axis=1)
    X = X.drop(['Geography', 'Gender'], axis=1)

    # add 3 new items
    X['BalanceSalaryRatio'] = X.Balance / X.EstimatedSalary
    # Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
    X['TenureByAge'] = X.Tenure / (X.Age)
    '''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
    :-)'''
    X['CreditScoreGivenAge'] = X.CreditScore / (X.Age)

    data_cols=X.columns.to_list()

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_pp = scaler.transform(X)
    X_f=pd.DataFrame(X_pp,columns=data_cols)

    # X_f.to_csv('data/X_data_bank_churn.csv', index=False)
    y.to_csv('data/y_data_bank_churn.csv', index=False)

def split_csv(x_file,y_file):
    X = pd.read_csv('data/X_data_bank_churn.csv', delimiter=',')
    y = pd.read_csv('data/y_data_bank_churn.csv', delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.head())
    print(y_train.head())
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)

    svm.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test

split_csv('data/X_data_bank_churn.csv','data/y_data_bank_churn.csv')






